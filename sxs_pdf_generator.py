import io
import os
import re
import json
import time
import base64
import tempfile
import traceback
from datetime import datetime
from typing import List, Optional, BinaryIO, Tuple, Dict, Any, Union

import streamlit as st
import requests
from PIL import Image, ImageDraw

# PDF Generation
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Page Configuration
PAGE_CONFIG = {
    "page_title": "SxS Model Comparison PDF Generator",
    "page_icon": "🖨️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Application Constants
class AppConfig:
    # Google Apps Script Integration
    WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxeqphIFWgbP5-71n5ws4w5JQOya_2q-RowAzHR9XoVY-0bPQ9M03gapkb4siEGBUHD/exec"
    #  st.secrets.get("webhook_url", "")
    WEBHOOK_TIMEOUT = 30
    
    # File Handling
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg']
    
    # PDF Configuration
    PDF_PAGE_WIDTH = 10 * inch  # Google Slides 16:9 format
    PDF_PAGE_HEIGHT = 5.625 * inch
    PDF_SAFE_MARGIN = 0.25 * inch
    
    # Security
    MAX_SESSION_DURATION = 3600
    MAX_EMAIL_ATTEMPTS = 10


class ModelBrandManager:
    """Manages brand recognition and color coding for model names."""
    
    # Brand recognition mapping - static reference for color coding
    BRAND_COLOR_MAP = {
        # Google/Gemini family
        "bard": {"color": "#4285f4", "brand": "Gemini"},
        "gemini": {"color": "#4285f4", "brand": "Gemini"}, 
        "ais": {"color": "#4285f4", "brand": "Google AI Studio"},
        "google": {"color": "#4285f4", "brand": "Google AI Studio"},
        
        # OpenAI family
        "chatgpt": {"color": "#10a37f", "brand": "OpenAI"},
        "cgpt": {"color": "#10a37f", "brand": "OpenAI"},
        "gpt": {"color": "#10a37f", "brand": "OpenAI"},
        "openai": {"color": "#10a37f", "brand": "OpenAI"},
        
        # Anthropic family
        "claude": {"color": "#d97706", "brand": "Anthropic"},
        "anthropic": {"color": "#d97706", "brand": "Anthropic"},
        
        # Meta family  
        "llama": {"color": "#1877f2", "brand": "Meta"},
        "meta": {"color": "#1877f2", "brand": "Meta"},
        
        # Default fallback
        "default": {"color": "#6b7280", "brand": "Unknown"}
    }
    
    @staticmethod
    @st.cache_data
    def get_model_config(model_name: str) -> Dict[str, str]:
        """Get color and brand info for a model name."""
        if not model_name:
            return {
                "color": ModelBrandManager.BRAND_COLOR_MAP["default"]["color"],
                "brand": ModelBrandManager.BRAND_COLOR_MAP["default"]["brand"],
                "logo_text": "Unknown"
            }
        
        model_lower = model_name.lower()
        
        # Check each brand keyword
        for brand_key, config in ModelBrandManager.BRAND_COLOR_MAP.items():
            if brand_key != "default" and brand_key in model_lower:
                return {
                    "color": config["color"],
                    "brand": config["brand"],
                    "logo_text": model_name
                }
        
        # Default fallback
        return {
            "color": ModelBrandManager.BRAND_COLOR_MAP["default"]["color"],
            "brand": ModelBrandManager.BRAND_COLOR_MAP["default"]["brand"], 
            "logo_text": model_name
        }
    
    @staticmethod
    @st.cache_data
    def get_brand_gradient(model1_name: str, model2_name: str) -> str:
        """Get CSS gradient for two model brands."""
        model1_config = ModelBrandManager.get_model_config(model1_name)
        model2_config = ModelBrandManager.get_model_config(model2_name)
        
        return f"linear-gradient(90deg, {model1_config['color']}22, {model2_config['color']}22)"

# ============================================================================
# BULLETPROOF BUTTON MANAGEMENT SYSTEM
# ============================================================================

class ButtonManager:
    """Centralized button state management with immediate locking to prevent race conditions."""
    
    @staticmethod
    def initialize_button_states():
        """Initialize all button states in session state."""
        button_defaults = {
            'drive_upload_state': 'ready',  # ready, processing, locked
            'form_submit_state': 'ready',   # ready, processing, locked
            'drive_upload_processing_started': False,
            'form_submit_processing_started': False,
        }
        
        for key, default_value in button_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def get_button_state(button_id: str) -> str:
        """Get current button state: ready, processing, or locked."""
        return st.session_state.get(f"{button_id}_state", "ready")
    
    @staticmethod
    def is_button_ready(button_id: str) -> bool:
        """Check if button is ready to be clicked."""
        return ButtonManager.get_button_state(button_id) == "ready"
    
    @staticmethod
    def is_button_processing(button_id: str) -> bool:
        """Check if button is currently processing."""
        return ButtonManager.get_button_state(button_id) == "processing"
    
    @staticmethod
    def is_button_locked(button_id: str) -> bool:
        """Check if button is permanently locked."""
        return ButtonManager.get_button_state(button_id) == "locked"
    
    @staticmethod
    def start_processing(button_id: str) -> bool:
        """Start processing for a button. Returns True if state change was successful."""
        current_state = ButtonManager.get_button_state(button_id)
        
        if current_state != "ready":
            return False
        
        # IMMEDIATELY set processing state
        st.session_state[f"{button_id}_state"] = "processing"
        st.session_state[f"{button_id}_processing_started"] = True
        
        return True
    
    @staticmethod
    def complete_processing(button_id: str, success: bool = True):
        """Complete processing and set final state."""
        if success:
            st.session_state[f"{button_id}_state"] = "locked"
        else:
            # Failed - reset to ready for retry
            st.session_state[f"{button_id}_state"] = "ready"
        
        st.session_state[f"{button_id}_processing_started"] = False
    
    @staticmethod
    def should_process(button_id: str) -> bool:
        """Check if we should process this button (processing state + flag set)."""
        return (ButtonManager.is_button_processing(button_id) and 
                st.session_state.get(f"{button_id}_processing_started", False))
    
    @staticmethod
    def reset_button(button_id: str):
        """Reset button to ready state (for debugging/admin use)."""
        st.session_state[f"{button_id}_state"] = "ready"
        st.session_state[f"{button_id}_processing_started"] = False
    
    @staticmethod
    def get_button_display_info(button_id: str) -> Dict[str, str]:
        """Get display information for button based on current state."""
        state = ButtonManager.get_button_state(button_id)
        
        display_map = {
            "drive_upload": {
                "ready": {"text": "📤 Upload", "type": "primary", "disabled": False},
                "processing": {"text": "⚙️ Uploading...", "type": "secondary", "disabled": True},
                "locked": {"text": "✅ Uploaded", "type": "secondary", "disabled": True}
            },
            "form_submit": {
                "ready": {"text": "📋 Submit Form", "type": "primary", "disabled": False},
                "processing": {"text": "⚙️ Submitting...", "type": "secondary", "disabled": True},
                "locked": {"text": "✅ Submitted", "type": "secondary", "disabled": True}
            }
        }
        
        return display_map.get(button_id, {}).get(state, {
            "text": f"{button_id} ({state})", 
            "type": "secondary", 
            "disabled": True
        })

# ============================================================================
# SECURITY & SESSION MANAGEMENT
# ============================================================================

class SecurityManager:
    """Enhanced security manager with proper button state integration."""
    
    @staticmethod
    def initialize_session_state() -> None:
        """Initialize all security-related session state variables."""
        security_defaults = {
            # Core workflow locks
            'submission_locked': False,
            'drive_upload_locked': False,
            'form_submitted': False,
            
            # Validation states
            'email_validated': False,
            'drive_url_generated': False,
            'pdf_generated': False,
            'question_id_validated': False,
            
            # Navigation and workflow
            'current_page': 'Metadata Input',
            'previous_page': None,
            'navigation_locked': False,
            
            # Session tracking
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            'session_start_time': datetime.now().timestamp(),
            'form_data_locked': False,
            
            # Timestamps
            'drive_upload_timestamp': None,
            'form_submission_timestamp': None,
        }
        
        for key, default_value in security_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Initialize button states
        ButtonManager.initialize_button_states()
    
    @staticmethod
    def can_upload_to_drive() -> bool:
        """Check if drive upload is allowed based on prerequisites and button state."""
        prerequisites_met = (
            st.session_state.get('email_validated', False) and 
            st.session_state.get('pdf_generated', False) and
            not st.session_state.get('drive_upload_locked', False)
        )
        
        button_ready = ButtonManager.is_button_ready('drive_upload')
        
        return prerequisites_met and button_ready
    
    @staticmethod
    def can_submit_form() -> bool:
        """Check if form submission is allowed based on prerequisites and button state."""
        # Check basic prerequisites (matching upload button logic)
        prerequisites_met = (
            st.session_state.get('email_validated', False) and
            st.session_state.get('drive_url_generated', False) and
            bool(st.session_state.get('drive_url', '')) and
            not st.session_state.get('form_submitted', False)
        )
        
        # Check button is ready to be clicked
        button_ready = ButtonManager.is_button_ready('form_submit')
        
        return prerequisites_met and button_ready
    
    @staticmethod
    def is_session_expired() -> bool:
        """Check if current session has expired."""
        if 'session_start_time' not in st.session_state:
            return True
        
        elapsed = datetime.now().timestamp() - st.session_state.session_start_time
        return elapsed > AppConfig.MAX_SESSION_DURATION
    
    @staticmethod
    def is_workflow_completed() -> bool:
        """Check if the entire workflow has been completed."""
        return st.session_state.get('form_submitted', False)
    
    @staticmethod
    def is_navigation_allowed(target_page: str) -> bool:
        """Check if navigation to target page is allowed."""
        if SecurityManager.is_workflow_completed():
            return target_page in ['Help', 'Upload to Drive']
        
        # Normal workflow progression rules
        navigation_rules = {
            'Image Upload': ['question_id', 'prompt_text', 'model1', 'model2'],
            'PDF Generation': ['model1_images', 'model2_images'],
            'Upload to Drive': ['pdf_generated'],
        }
        
        required_keys = navigation_rules.get(target_page, [])
        return all(key in st.session_state for key in required_keys)
    
    @staticmethod
    def reset_session() -> None:
        """Reset all session state for a new session."""
        # Keep only essential keys
        essential_keys = ['current_page']
        keys_to_clear = [key for key in st.session_state.keys() if key not in essential_keys]
        
        for key in keys_to_clear:
            del st.session_state[key]
        
        # Reinitialize
        SecurityManager.initialize_session_state()

# ============================================================================
# GOOGLE APPS SCRIPT INTEGRATION
# ============================================================================

class AppsScriptClient:
    """Secure client for Google Apps Script webhook integration."""
    
    def __init__(self):
        self.webhook_url = AppConfig.WEBHOOK_URL
    
    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a secure request to the webhook."""
        try:
            if not self.webhook_url:
                return {"success": False, "message": "Webhook URL not configured"}
            
            # Add session tracking
            data['session_id'] = st.session_state.get('session_id', 'unknown')
            
            response = requests.post(
                self.webhook_url,
                json=data,
                timeout=AppConfig.WEBHOOK_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "message": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "message": f"Request error: {str(e)}"}
    
    def validate_email(self, email: str, attempt_count: int = 1) -> Dict[str, Any]:
        """Validate email against authorized list."""
        if SecurityManager.is_workflow_completed():
            return {"success": False, "message": "Workflow completed - validation locked"}
        
        return self._make_request({
            "action": "validate_email",
            "email": email,
            "attempt_count": attempt_count
        })
    
    def validate_question_id(self, question_id: str) -> Dict[str, Any]:
        """Validate Question ID against SOT spreadsheet."""
        if SecurityManager.is_workflow_completed():
            return {"success": False, "message": "Workflow completed - validation locked"}
        
        return self._make_request({
            "action": "validate_question_id",
            "question_id": question_id
        })
    
    def upload_pdf(self, pdf_buffer: io.BytesIO, filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Upload PDF to Google Drive."""
        # Validate file size
        pdf_buffer.seek(0)
        pdf_data = pdf_buffer.read()
        
        if len(pdf_data) > AppConfig.MAX_FILE_SIZE_BYTES:
            return {"success": False, "message": f"File too large (max {AppConfig.MAX_FILE_SIZE_MB}MB)"}
        
        # Prepare upload data
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # Make the upload request
        result = self._make_request({
            "action": "upload_pdf",
            "pdf_base64": pdf_base64,
            "filename": filename,
            "metadata": metadata
        })
        
        return result
    
    def log_submission(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log form submission to spreadsheet."""
        result = self._make_request({
            "action": "log_submission",
            **form_data
        })
        
        return result

# ============================================================================
# CACHING STRATEGIES
# ============================================================================

@st.cache_resource
def get_apps_script_client() -> AppsScriptClient:
    """Get cached AppsScript client instance."""
    return AppsScriptClient()

@st.cache_data
def validate_email_format(email: str) -> bool:
    """Validate email format using regex."""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

@st.cache_data
def extract_task_id_from_question_id(question_id: str) -> Optional[str]:
    """Extract Task ID from Question ID using pattern matching."""
    try:
        # Primary pattern
        pattern = r'bard_data\+([^+]+)\+INTERNAL'
        match = re.search(pattern, question_id)
        
        if match and match[1]:
            return match[1].strip()
        
        # Fallback patterns
        alt_patterns = [
            r'bard_data\+([^+]+)\+',
            r'coach_P\d+[^+]+',
        ]
        
        for alt_pattern in alt_patterns:
            alt_match = re.search(alt_pattern, question_id)
            if alt_match:
                return alt_match[0].replace('bard_data+', '').replace('+', '').strip()
        
        return None
    except Exception:
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class Utils:
    """Utility functions for the application."""
    
    @staticmethod
    def sanitize_html_output(text: str) -> str:
        """Sanitize text for safe HTML output."""
        import html
        return html.escape(str(text))
    
    @staticmethod
    def validate_file_size(file) -> bool:
        """Validate uploaded file size."""
        if hasattr(file, 'size'):
            return file.size <= AppConfig.MAX_FILE_SIZE_BYTES
        return True
    
    @staticmethod
    def generate_filename(model1: str, model2: str) -> str:
        """Generate standardized filename for PDF with brand awareness."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean model names for filename
        model1_clean = re.sub(r'[^\w\-_.]', '_', model1)
        model2_clean = re.sub(r'[^\w\-_.]', '_', model2)
        
        session_id = st.session_state.get('session_id', 'unknown')[:8]
        
        return f"SxS_Comparison_{model1_clean}_vs_{model2_clean}_{timestamp}_{session_id}.pdf"
    
    @staticmethod
    def parse_question_id(question_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse Question ID to extract language and project type."""
        language = None
        project_type = None
        
        try:
            # Extract language
            language_pattern = r'human_eval_([a-z]{2}-[A-Z]{2}|[a-z]{2}-\d{3}|[a-z]{2}-[a-z]{2})\+INTERNAL'
            language_match = re.search(language_pattern, question_id)
            
            if language_match:
                language = language_match.group(1)
            
            # Extract project type
            project_type_mapping = {
                'monolingual': 'Monolingual',
                'audio_out': 'Audio Out',
                'mixed': 'Mixed',
                'code_mixed': 'Mixed',
                'language_learning': 'Language Learning',
                'learning_and_academic_help': 'Learning & Academic Help'
            }
            
            project_pattern = r'experience_([a-z_]+)_human_eval'
            project_match = re.search(project_pattern, question_id)
            
            if project_match:
                extracted_project = project_match.group(1)
                for key, value in project_type_mapping.items():
                    if key in extracted_project:
                        project_type = value
                        break
        
        except Exception as e:
            st.error(f"Error parsing Question ID: {e}")
        
        return language, project_type
    
    @staticmethod
    @st.cache_data
    def parse_model_combination(model_comparison: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse model combination string into individual models with better matching."""
        try:
            if not model_comparison:
                return None, None
            
            # Try multiple separator patterns
            separators = [" vs. ", " vs ", " v. ", " v ", " VS. ", " VS "]
            
            for separator in separators:
                if separator in model_comparison:
                    parts = model_comparison.split(separator, 1)  # Split only on first occurrence
                    if len(parts) == 2:
                        return parts[0].strip(), parts[1].strip()
            
            return None, None
        except Exception:
            return None, None

# ============================================================================
# PDF GENERATION CLASS
# ============================================================================

class PDFGenerator:
    """Production-grade PDF generator with Google Slides format and brand awareness."""
    
    def __init__(self):
        self.page_width = AppConfig.PDF_PAGE_WIDTH
        self.page_height = AppConfig.PDF_PAGE_HEIGHT
        self.slide_format = (self.page_width, self.page_height)
        self.safe_margin = AppConfig.PDF_SAFE_MARGIN
        self.content_width = self.page_width - (2 * self.safe_margin)
        self.content_height = self.page_height - (2 * self.safe_margin)
        
        # Colors
        self.primary_color = HexColor('#4a86e8')
        self.text_color = HexColor('#1f2937')
        self.light_gray = HexColor('#f3f4f6')
        
        self.temp_files = []
        self.company_logo_path = None
        self._setup_company_logo()
    
    def _setup_company_logo(self):
        """Create company logo for PDF."""
        try:
            logo_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            
            icon_size = 72
            logo_img = Image.new('RGBA', (icon_size, icon_size), (255, 255, 255, 0))
            draw = ImageDraw.Draw(logo_img)
            
            circle_margin = 4
            circle_size = icon_size - (2 * circle_margin)
            
            # Draw logo elements
            draw.ellipse([circle_margin, circle_margin, 
                         circle_margin + circle_size, circle_margin + circle_size], 
                        fill=(15, 15, 15, 255), outline=None)
            
            inner_margin = 12
            inner_size = circle_size - (2 * inner_margin)
            inner_x = circle_margin + inner_margin
            inner_y = circle_margin + inner_margin
            
            draw.rectangle([inner_x, inner_y, inner_x + inner_size, inner_y + inner_size], 
                         fill=(255, 255, 255, 255))
            
            logo_img.save(logo_temp.name, format='PNG')
            logo_temp.close()
            
            self.company_logo_path = logo_temp.name
            self.temp_files.append(logo_temp.name)
            
        except Exception as e:
            st.warning(f"Could not create company logo: {e}")
            self.company_logo_path = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                st.warning(f"Could not clean up temp file: {e}")
        self.temp_files = []
    
    def prepare_image(self, image_file: BinaryIO) -> Optional[str]:
        """Convert uploaded image to ReportLab compatible format."""
        try:
            image_file.seek(0)
            img = Image.open(image_file)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(temp_file.name, format='JPEG', quality=95, optimize=True)
            temp_file.close()
            
            self.temp_files.append(temp_file.name)
            return temp_file.name
            
        except Exception as e:
            st.error(f"Error preparing image: {str(e)}")
            return None
    
    def draw_company_logo(self, canvas_obj):
        """Draw company logo on PDF."""
        if not self.company_logo_path:
            return
        
        try:
            logo_size = 0.5 * inch
            logo_margin = 0.2 * inch
            logo_x = self.page_width - logo_size - logo_margin
            logo_y = logo_margin
            
            canvas_obj.drawImage(
                self.company_logo_path,
                logo_x, logo_y,
                width=logo_size, height=logo_size,
                preserveAspectRatio=True
            )
        except Exception as e:
            st.warning(f"Could not draw company logo: {e}")
    
    def draw_slide_background(self, canvas_obj):
        """Draw slide background."""
        canvas_obj.setFillColor(HexColor('#ffffff'))
        canvas_obj.rect(0, 0, self.page_width, self.page_height, fill=1, stroke=0)
        
        canvas_obj.setStrokeColor(HexColor('#e5e7eb'))
        canvas_obj.setLineWidth(1)
        canvas_obj.rect(0, 0, self.page_width, self.page_height, fill=0, stroke=1)
    
    def draw_wrapped_text(self, canvas_obj, text: str, x: float, y: float, 
                         max_width: float, font_name: str = "Helvetica", 
                         font_size: int = 12, line_height_factor: float = 1.2) -> float:
        """Draw text with automatic line wrapping."""
        canvas_obj.setFont(font_name, font_size)
        canvas_obj.setFillColor(self.text_color)
        
        def break_long_word(word, max_word_width):
            if canvas_obj.stringWidth(word, font_name, font_size) <= max_word_width:
                return [word]
            
            broken_parts = []
            current_part = ""
            
            for char in word:
                test_part = current_part + char
                if canvas_obj.stringWidth(test_part, font_name, font_size) <= max_word_width:
                    current_part = test_part
                else:
                    if current_part:
                        broken_parts.append(current_part)
                    current_part = char
            
            if current_part:
                broken_parts.append(current_part)
            
            return broken_parts
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if canvas_obj.stringWidth(word, font_name, font_size) > max_width:
                if current_line.strip():
                    lines.append(current_line.strip())
                    current_line = ""
                
                broken_words = break_long_word(word, max_width)
                for i, broken_word in enumerate(broken_words):
                    if i == len(broken_words) - 1:
                        current_line = broken_word + " "
                    else:
                        lines.append(broken_word)
            else:
                test_line = current_line + word + " "
                if canvas_obj.stringWidth(test_line, font_name, font_size) <= max_width:
                    current_line = test_line
                else:
                    if current_line.strip():
                        lines.append(current_line.strip())
                    current_line = word + " "
        
        if current_line.strip():
            lines.append(current_line.strip())
        
        current_y = y
        line_height = font_size * line_height_factor
        
        for line in lines:
            canvas_obj.drawString(x, current_y, line)
            current_y -= line_height
        
        return current_y
    
    def draw_centered_text(self, canvas_obj, text: str, y: float, 
                          font_name: str = "Helvetica-Bold", font_size: int = 48,
                          color: HexColor = None):
        """Draw centered text."""
        if color is None:
            color = self.text_color
        
        canvas_obj.setFont(font_name, font_size)
        canvas_obj.setFillColor(color)
        
        text_width = canvas_obj.stringWidth(text, font_name, font_size)
        x = (self.page_width - text_width) / 2
        canvas_obj.drawString(x, y, text)
    
    def draw_image_centered(self, canvas_obj, image_path: str, max_width: float = None, 
                           max_height: float = None):
        """Draw image centered on slide."""
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            if max_width is None:
                max_width = self.content_width
            if max_height is None:
                max_height = self.content_height
            
            if img_width > max_width or img_height > max_height:
                ratio = min(max_width / img_width, max_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
            else:
                new_width = img_width
                new_height = img_height
            
            x = (self.page_width - new_width) / 2
            y = (self.page_height - new_height) / 2
            
            canvas_obj.drawImage(image_path, x, y, width=new_width, height=new_height)
            
        except Exception as e:
            st.error(f"Error drawing image: {str(e)}")
    
    def create_title_slide(self, canvas_obj, question_id: str, prompt: str, 
                          prompt_image: Optional[BinaryIO] = None):
        """Create title slide with question ID and prompt."""
        self.draw_slide_background(canvas_obj)
        
        y_pos = self.page_height - self.safe_margin - 15
        
        # Question ID section
        canvas_obj.setFont("Helvetica-Bold", 14)
        canvas_obj.setFillColor(self.primary_color)
        canvas_obj.drawString(self.safe_margin, y_pos, "ID:")
        y_pos -= 18
        
        y_pos = self.draw_wrapped_text(canvas_obj, question_id, 
                                     self.safe_margin, y_pos, 
                                     self.content_width, 
                                     font_name="Helvetica", font_size=9,
                                     line_height_factor=1.1)
        y_pos -= 25
        
        # Layout for prompt text and image
        if prompt_image is not None:
            text_column_width = self.content_width * 0.60
            gap_width = self.content_width * 0.02
            image_column_width = self.content_width * 0.38
            image_column_x = self.safe_margin + text_column_width + gap_width
        else:
            text_column_width = self.content_width
            image_column_width = 0
            image_column_x = 0
        
        content_start_y = y_pos
        
        # Prompt text
        canvas_obj.setFont("Helvetica-Bold", 14)
        canvas_obj.setFillColor(self.primary_color)
        canvas_obj.drawString(self.safe_margin, y_pos, "Initial Prompt:")
        y_pos -= 20
        
        self.draw_wrapped_text(canvas_obj, prompt,
                              self.safe_margin, y_pos,
                              text_column_width,
                              font_name="Helvetica", font_size=12,
                              line_height_factor=1.3)
        
        # Prompt image
        if prompt_image is not None:
            available_height = content_start_y - self.safe_margin - 60
            self.draw_prompt_image_in_column(canvas_obj, prompt_image,
                                           image_column_x, content_start_y - 20,
                                           image_column_width, available_height)
        
        self.draw_company_logo(canvas_obj)
    
    def draw_prompt_image_in_column(self, canvas_obj, image_file: BinaryIO, 
                                   x: float, y: float, column_width: float, 
                                   available_height: float):
        """Draw prompt image within column bounds."""
        try:
            image_file.seek(0)
            image_data = image_file.read()
            
            if not image_data:
                return
            
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_image.write(image_data)
            temp_image.close()
            
            self.temp_files.append(temp_image.name)
            
            img = Image.open(temp_image.name)
            img_width, img_height = img.size
            
            width_ratio = column_width / img_width
            height_ratio = available_height / img_height
            scale_ratio = min(width_ratio, height_ratio, 1.0)
            
            new_width = img_width * scale_ratio
            new_height = img_height * scale_ratio
            
            image_x = x + (column_width - new_width) / 2
            image_y = y - new_height
            
            min_y = self.safe_margin + 60
            if image_y < min_y:
                adjusted_height = y - min_y
                height_ratio = adjusted_height / img_height
                scale_ratio = min(width_ratio, height_ratio, 1.0)
                
                new_width = img_width * scale_ratio
                new_height = img_height * scale_ratio
                image_x = x + (column_width - new_width) / 2
                image_y = y - new_height
            
            canvas_obj.drawImage(temp_image.name, image_x, image_y, 
                               width=new_width, height=new_height,
                               preserveAspectRatio=True)
            
        except Exception as e:
            st.error(f"Error drawing prompt image: {str(e)}")
    
    def create_model_title_slide(self, canvas_obj, model_name: str):
        """Create model title slide with brand-aware styling."""
        self.draw_slide_background(canvas_obj)
        
        # Get brand configuration
        model_config = ModelBrandManager.get_model_config(model_name)
        brand_color = HexColor(model_config['color'])
        
        self.draw_centered_text(
            canvas_obj, 
            model_name, 
            self.page_height / 2, 
            font_name="Helvetica-Bold", 
            font_size=56,
            color=brand_color
        )
        
        # Add brand subtitle
        self.draw_centered_text(
            canvas_obj,
            model_config['brand'],
            self.page_height / 2 - 80,
            font_name="Helvetica",
            font_size=24,
            color=self.text_color
        )
        
        self.draw_company_logo(canvas_obj)
    
    def create_image_slide(self, canvas_obj, image_path: str):
        """Create image slide."""
        self.draw_slide_background(canvas_obj)
        
        max_height = self.content_height - 20
        max_width = self.content_width - 20
        
        self.draw_image_centered(canvas_obj, image_path, 
                               max_width=max_width, 
                               max_height=max_height)
        
        self.draw_company_logo(canvas_obj)
    
    def generate_pdf(self, question_id: str, prompt: str, model1: str, model2: str,
                    model1_images: List[BinaryIO], model2_images: List[BinaryIO],
                    prompt_image: Optional[BinaryIO] = None) -> io.BytesIO:
        """Generate the complete PDF."""
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=self.slide_format)
        
        try:
            # Title slide
            self.create_title_slide(c, question_id, prompt, prompt_image)
            
            # First model
            c.showPage()
            self.create_model_title_slide(c, model1)
            
            for img_file in model1_images:
                c.showPage()
                temp_image_path = self.prepare_image(img_file)
                if temp_image_path:
                    self.create_image_slide(c, temp_image_path)
            
            # Second model
            c.showPage()
            self.create_model_title_slide(c, model2)
            
            for img_file in model2_images:
                c.showPage()
                temp_image_path = self.prepare_image(img_file)
                if temp_image_path:
                    self.create_image_slide(c, temp_image_path)
            
            c.save()
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            raise e

# ============================================================================
# UI COMPONENTS & STYLING
# ============================================================================

class UIComponents:
    """Reusable UI components and styling."""
    
    @staticmethod
    def load_custom_css():
        """Load custom CSS styles."""
        st.markdown("""
        <style>
            .main-header {
                text-align: center;
                padding: 1rem 0;
                background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 1rem;
            }
            
            .step-indicator {
                display: flex;
                justify-content: space-between;
                margin: 2rem 0;
                padding: 1rem;
                background-color: #16213e;
                border-radius: 10px;
                color: white;
            }
            
            .step {
                text-align: center;
                padding: 0.5rem;
                border-radius: 5px;
                font-weight: bold;
                flex: 1;
                margin: 0 0.25rem;
            }
            
            .step.active {
                background-color: #4285f4;
                color: white;
            }
            
            .step.completed {
                background-color: #34a853;
                color: white;
            }
            
            .step.locked {
                background-color: #dc3545;
                color: white;
                opacity: 0.7;
            }
            
            .security-banner {
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                margin: 1rem 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .security-warning {
                background: linear-gradient(135deg, #dc3545, #fd7e14);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                margin: 1rem 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .upload-section {
                border: 2px dashed #71b280;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                margin: 1rem 0;
                background-color: #DCF58F;
                color: black;
            }
            
            .upload-section.locked {
                border: 2px dashed #dc3545;
                background-color: #f8d7da;
                color: #721c24;
                opacity: 0.6;
            }
            
            .success-message {
                background-color: #d4edda;
                color: #155724;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
                border: 1px solid #c3e6cb;
            }
            
            .error-message {
                background-color: #f8d7da;
                color: #721c24;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
                border: 1px solid #f5c6cb;
            }
            
            .info-card {
                background-color: #e3f2fd;
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 4px solid #2196f3;
                margin: 1rem 0;
                color: #0d47a1;
            }
            
            .stats-container {
                display: flex;
                justify-content: space-around;
                margin: 2rem 0;
            }
            
            .stat-card {
                text-align: center;
                padding: 1rem;
                background-color: #DCF58F;
                color: black;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                flex: 1;
                margin: 0 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_workflow_completion_status():
        """Display workflow completion status."""
        if SecurityManager.is_workflow_completed():
            st.markdown(f"""
            <div class="security-banner">
                <h2>🎉 WORKFLOW COMPLETED SUCCESSFULLY</h2>
                <p><strong>Session ID:</strong> {st.session_state.get('session_id', 'N/A')}</p>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 1.5rem 0;">
                <p><strong>📋 SUMMARY:</strong></p>
                <p>✅ PDF Generated & Downloaded</p>
                <p>✅ Uploaded to Google Drive</p>
                <p>✅ Form Submitted to Tracking System</p>
                <br>
                <p><em>🔒 This session is now locked. Start a new session for another comparison.</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def display_step_indicator(current_page: str):
        """Display step progress indicator."""
        steps = ["1️⃣ Metadata Input", "2️⃣ Image Upload", "3️⃣ PDF Generation", "4️⃣ Upload to Drive"]
        
        if current_page == "Help":
            return
        
        statuses = UIComponents.get_step_status(current_page)
        
        step_html = '<div class="step-indicator">'
        for step, status in zip(steps, statuses):
            step_html += f'<div class="step {status}">{step}</div>'
        step_html += '</div>'
        
        st.markdown(step_html, unsafe_allow_html=True)
    
    @staticmethod
    def get_step_status(current_page: str) -> List[str]:
        """Get status of each workflow step."""
        steps = ["1️⃣ Metadata Input", "2️⃣ Image Upload", "3️⃣ PDF Generation", "4️⃣ Upload to Drive"]
        statuses = []
        
        workflow_completed = SecurityManager.is_workflow_completed()
        
        for step in steps:
            if workflow_completed and not step.endswith(current_page):
                statuses.append("locked")
            elif step.endswith(current_page):
                statuses.append("active")
            elif UIComponents.is_step_completed(step.split(" ", 1)[1]):
                statuses.append("completed")
            else:
                statuses.append("")
        
        return statuses
    
    @staticmethod
    def is_step_completed(step_name: str) -> bool:
        """Check if a workflow step is completed."""
        # Help is not a completable step
        if step_name == "Help":
            return False
            
        completion_checks = {
            "Metadata Input": ['question_id', 'prompt_text', 'model1', 'model2'],
            "Image Upload": ['model1_images', 'model2_images'],
            "PDF Generation": ['pdf_buffer'],
            "Upload to Drive": ['form_submitted'],
        }
        
        required_keys = completion_checks.get(step_name, [])
        
        if step_name == "Upload to Drive":
            return st.session_state.get('form_submitted', False)
        
        return all(key in st.session_state for key in required_keys)

# ============================================================================
# EMAIL VALIDATION SYSTEM
# ============================================================================

class EmailValidator:
    """Updated email validator - simplified attempt tracking."""
    
    @staticmethod
    def validate_with_attempts(email: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate email with simplified attempt tracking."""
        if SecurityManager.is_workflow_completed():
            return False, "❌ Workflow already completed - validation locked", {}
        
        if not email or not email.strip():
            return False, "Email is required", {}
        
        if not validate_email_format(email):
            return False, "Invalid email format", {}
        
        # Track attempts for abuse prevention
        email_key = f"email_attempts_{email.lower().strip()}"
        current_attempts = st.session_state.get(email_key, 0) + 1
        st.session_state[email_key] = current_attempts
        
        # Prevent excessive attempts
        if current_attempts > AppConfig.MAX_EMAIL_ATTEMPTS:
            return False, f"❌ Too many attempts ({current_attempts}). Please refresh the page.", {}
        
        try:
            apps_script = get_apps_script_client()
            validation_result = apps_script.validate_email(email.strip(), current_attempts)
            
            if validation_result.get("success"):
                validation_data = validation_result.get("data", {})
                validation_type = validation_data.get("validation_type", "unknown")
                
                # Simplified validation - no company fallback
                message_map = {
                    "alias_list": "✅ Email found in authorized alias list",
                    "company_fallback": "✅ Company email accepted",
                }
                
                message = message_map.get(validation_type, "✅ Email validated successfully")
                return True, message, validation_data
            else:
                error_message = validation_result.get("message", "Email validation failed")
                validation_data = validation_result.get("data", {})
                return False, f"❌ {error_message}", validation_data
                
        except Exception as e:
            return False, f"⚠️ Email validation error: {str(e)}", {}
    
    @staticmethod
    def reset_attempts(email: str):
        """Reset attempt count for specific email."""
        email_key = f"email_attempts_{email.lower().strip()}"
        if email_key in st.session_state:
            del st.session_state[email_key]
    
    @staticmethod
    def get_attempt_count(email: str) -> int:
        """Get current attempt count for email."""
        email_key = f"email_attempts_{email.lower().strip()}"
        return st.session_state.get(email_key, 0)

# ============================================================================
# FORM VALIDATION & PROCESSING
# ============================================================================

class FormProcessor:
    """Handle form processing and validation with proper separation of concerns."""
    
    @staticmethod
    def validate_question_id(question_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate Question ID against SOT."""
        try:
            apps_script = get_apps_script_client()
            validation_result = apps_script.validate_question_id(question_id)
            
            if validation_result.get("success"):
                data = validation_result.get("data", {})
                is_valid = data.get("is_valid", True)
                message = "Question ID validated successfully" if is_valid else "Question ID not found in SOT"
                return is_valid, message, data
            else:
                return True, "Question ID accepted", {}
                
        except Exception as e:
            st.error(f"Validation error: {str(e)}")
            return True, "Question ID accepted", {}
    
    @staticmethod
    def process_drive_upload(user_email: str, filename: str) -> Dict[str, Any]:
        """Process PDF upload to Google Drive - ACTUAL PROCESSING ONLY."""
        try:
            apps_script = get_apps_script_client()
            
            metadata = {
                'user_email': user_email,
                'question_id': st.session_state.question_id,
                'model1': st.session_state.model1,
                'model2': st.session_state.model2,
                'timestamp': datetime.now().isoformat(),
                'session_id': st.session_state.get('session_id', 'unknown')
            }
            
            # Show progress indicator
            progress_placeholder = st.empty()
            progress_placeholder.info("📤 Uploading PDF to Google Drive...")
            
            # Perform actual upload
            upload_result = apps_script.upload_pdf(st.session_state.pdf_buffer, filename, metadata)
            
            progress_placeholder.empty()
            
            if upload_result.get("success"):
                # Extract drive URL with multiple fallback methods
                drive_url = None
                
                # Method 1: Standard structure
                if upload_result.get("data", {}).get("drive_url"):
                    drive_url = upload_result["data"]["drive_url"]
                
                # Method 2: Direct in response
                elif upload_result.get("drive_url"):
                    drive_url = upload_result["drive_url"]
                
                # Method 3: Check for shareable_url
                elif upload_result.get("data", {}).get("shareable_url"):
                    drive_url = upload_result["data"]["shareable_url"]
                
                if drive_url:
                    # Update session state with results
                    st.session_state.drive_url = drive_url
                    st.session_state.drive_url_generated = True
                    st.session_state.drive_upload_locked = True
                    st.session_state.drive_upload_timestamp = datetime.now().isoformat()
                    
                    return {
                        'success': True,
                        'drive_url': drive_url,
                        'message': 'Upload successful'
                    }
                else:
                    return {
                        'success': False,
                        'error_message': 'Drive URL not found in response',
                        'full_response': upload_result
                    }
            else:
                error_message = upload_result.get('message', 'Unknown API error')
                return {
                    'success': False,
                    'error_message': error_message
                }
                
        except Exception as e:
            return {
                'success': False,
                'error_message': f"Exception during upload: {str(e)}"
            }
    
    @staticmethod
    def process_form_submission(user_email: str, filename: str, file_size_kb: float) -> Dict[str, Any]:
        """Process final form submission - ACTUAL PROCESSING ONLY."""
        try:
            # ADDITIONAL SAFEGUARD: Double-check if already submitted
            if st.session_state.get('form_submitted', False):
                return {
                    'success': False,
                    'error_message': 'Form already submitted - duplicate submission prevented'
                }
            
            # ADDITIONAL SAFEGUARD: Check button state
            if not ButtonManager.is_button_processing('form_submit'):
                return {
                    'success': False,
                    'error_message': 'Invalid submission state - button not in processing mode'
                }
            
            apps_script = get_apps_script_client()
            
            form_data = {
                'user_email': user_email,
                'drive_url': st.session_state.get('drive_url', ''),
                'question_id': st.session_state.question_id,
                'language': st.session_state.get('sot_language', ''),
                'project_type': st.session_state.get('sot_project_type', ''),
                'prompt_text': st.session_state.prompt_text,
                'has_prompt_image': bool(st.session_state.get('prompt_image')),
                'model1': st.session_state.model1,
                'model1_image_count': len(st.session_state.model1_images),
                'model2': st.session_state.model2,
                'model2_image_count': len(st.session_state.model2_images),
                'pdf_filename': filename,
                'file_size_kb': file_size_kb,
                'session_id': st.session_state.get('session_id', 'unknown')
            }
            
            # Show progress indicator
            progress_placeholder = st.empty()
            progress_placeholder.info("📋 Submitting form to tracking system...")
            
            # Perform actual submission
            result = apps_script.log_submission(form_data)
            
            progress_placeholder.empty()
            
            if result.get("success"):
                # IMMEDIATELY mark as submitted to prevent any race conditions
                st.session_state.form_submitted = True
                
                # Clean up email attempts on success
                EmailValidator.reset_attempts(user_email)
                
                return {
                    'success': True,
                    'message': 'Form submitted successfully'
                }
            else:
                return {
                    'success': False,
                    'error_message': result.get('message', 'Unknown error')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error_message': f"Submission error: {str(e)}"
            }

# ============================================================================
# PAGE IMPLEMENTATIONS
# ============================================================================

class PageManager:
    """Manage individual page implementations."""
    
    @staticmethod
    def render_metadata_input():
        """Render metadata input page with integrated email validation."""
        st.header("1️⃣ Metadata Input")
        
        if SecurityManager.is_workflow_completed():
            UIComponents.display_workflow_completion_status()
            return
        
        st.markdown("""
        <div class="info-card">
            <h4>📋 Required Information</h4>
            <p>Please provide the following details for your model comparison task:<br></p>
            - Locate the <strong>Question ID</strong> at the top right of your CrC task — look for the 🛈 icon and refer to the "Question ID(s)" section (<a href="https://www.loom.com/share/7085c6bfa96149f7b443dd1f04f6de51?sid=250e5b7b-7a5c-4e64-92ff-0657c65567a9" target="_blank">see video guide</a>).<br>
            - Enter the alias email you use for work on the Google Crowd Compute (CrC) platform.<br>
            - Paste the exact task prompt and image prompt (if any) used during the side-by-side LLM evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        form_data = {}
        
        with st.form("metadata_form"):
            col1, col2 = st.columns([1, 1])
            
            # === COLUMN 1: Question ID & Email ===
            with col1:
                st.markdown("**🔍 Identification**")
                
                form_data['question_id'] = st.text_input(
                    "Question ID *",
                    placeholder="e.g.,  bfdf67160ca3eca9b65f040e350b2f1f+bard_data+coach_P128628...",
                    help="Enter the unique identifier for this comparison",
                    value=st.session_state.get('question_id', ''),
                    disabled=st.session_state.get('form_data_locked', False)
                )
                
                form_data['user_email'] = st.text_input(
                    "Alias Email Address *",
                    placeholder="e.g.,   ops-chiron...@invisible.co", 
                    help="Enter your CrC alias email address",
                    value=st.session_state.get('user_email', ''),
                    disabled=st.session_state.get('form_data_locked', False)
                )
            
            # === COLUMN 2: Prompt & Image ===
            with col2:
                
                st.markdown("**📝 Task Content**")
                
                form_data['prompt_text'] = st.text_area(
                    "Initial Prompt *",
                    placeholder="Enter the prompt used for both models...",
                    height=150,
                    help="What was the LLM tasked with doing, according to CrC?",
                    value=st.session_state.get('prompt_text', ''),
                    disabled=st.session_state.get('form_data_locked', False)
                )
                
                form_data['prompt_image'] = st.file_uploader(
                    "Prompt Image (Optional)",
                    type=AppConfig.SUPPORTED_IMAGE_TYPES,
                    help="Upload an image if the prompt included visual content",
                    disabled=st.session_state.get('form_data_locked', False)
                )
                
                if form_data['prompt_image'] and not Utils.validate_file_size(form_data['prompt_image']):
                    st.error(f"Prompt image is too large (max {AppConfig.MAX_FILE_SIZE_MB}MB)")
                    form_data['prompt_image'] = None
            
            # Form submit button (inside form)
            submitted = st.form_submit_button(
                "💾 Save Metadata", 
                type="primary",
                disabled=st.session_state.get('form_data_locked', False)
            )
        
        # === PROCESS FORM SUBMISSION (Outside form) ===
        if submitted:
            validation_results = PageManager._process_step1_validation(form_data)
            
            if validation_results.get('overall_success'):
                # === VALIDATION STATUS DISPLAY ===
                PageManager._display_step1_validation_status(form_data)
                PageManager._display_step1_success_popups(validation_results)
        
        # === NEXT STEP BUTTON ===
        if UIComponents.is_step_completed("Metadata Input"):
            PageManager._show_next_step_button("Metadata Input")
    
    @staticmethod
    def _process_step1_validation(form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate all Step 1 form data."""
        results = {
            'question_id_valid': False,
            'email_valid': False,
            'overall_success': False,
            'question_id_data': {},
            'email_data': {},
            'matched_models': None
        }
        
        # Check required fields
        if not all([form_data.get('question_id'), form_data.get('user_email'), form_data.get('prompt_text')]):
            st.error("❌ Please fill in all required fields marked with *.")
            return results
        
        # Validate Question ID
        if form_data['question_id']:
            qid_valid, qid_message, qid_data = FormProcessor.validate_question_id(form_data['question_id'])
            results['question_id_valid'] = qid_valid
            results['question_id_message'] = qid_message
            results['question_id_data'] = qid_data
            
            # Parse model combination from SOT data if available
            if qid_valid and qid_data.get('model_comparison'):
                sot_model1, sot_model2 = Utils.parse_model_combination(qid_data['model_comparison'])
                if sot_model1 and sot_model2:
                    results['matched_models'] = (sot_model1, sot_model2)
        
        # Validate Email
        if form_data['user_email']:
            email_valid, email_message, email_data = EmailValidator.validate_with_attempts(form_data['user_email'])
            results['email_valid'] = email_valid
            results['email_message'] = email_message
            results['email_data'] = email_data
        
        # Check if Question ID validation failed
        if not results['question_id_valid']:
            st.error(f"❌ Question ID Validation Failed: {results.get('question_id_message', 'Invalid Question ID')}")
            if results['question_id_data'].get('task_id'):
                st.error(f"Extracted Task ID: {results['question_id_data']['task_id']}")
            return results
        
        # Check if email validation failed
        if not results['email_valid']:
            st.error(f"❌ Email Validation Failed: {results.get('email_message', 'Invalid email')}")
            return results
        
        # Overall success check
        results['overall_success'] = results['question_id_valid'] and results['email_valid']
        
        if results['overall_success']:
            # Store validated data in session state
            session_updates = {
                'question_id': form_data['question_id'],
                'user_email': form_data['user_email'],
                'prompt_text': form_data['prompt_text'],
                'question_id_validated': True,
                'email_validated': True,
                'task_id': results['question_id_data'].get('task_id'),
                'sot_language': results['question_id_data'].get('language', ''),
                'sot_project_type': results['question_id_data'].get('project_type', ''),
                'sot_model_comparison': results['question_id_data'].get('model_comparison', ''),
            }
            
            # Set model information from SOT
            if results['matched_models']:
                session_updates['model1'] = results['matched_models'][0]
                session_updates['model2'] = results['matched_models'][1]
            else:
                # If no SOT match, this indicates invalid Question ID
                st.error("❌ Question ID is not valid - no model combination found in SOT")
                return results
            
            # Store prompt image if provided
            if form_data['prompt_image']:
                session_updates['prompt_image'] = form_data['prompt_image']
            
            # Update session state
            for key, value in session_updates.items():
                st.session_state[key] = value
        
        return results
    
    @staticmethod
    def _display_step1_validation_status(form_data: Dict[str, Any]):
        """Display unified validation status below the form."""
        if not (form_data.get('question_id') or form_data.get('user_email')):
            return
        
        # Get current validation states from session
        qid_validated = st.session_state.get('question_id_validated', False)
        email_validated = st.session_state.get('email_validated', False)
        
        if qid_validated and email_validated:
            st.success("✅ **All validations passed** - Question ID and Email are both verified")
        else:
            validation_status = []
            
            if form_data.get('question_id'):
                if qid_validated:
                    validation_status.append("✅ Question ID: Valid")
                else:
                    validation_status.append("❌ Question ID: Needs validation")
            
            if form_data.get('user_email'):
                if email_validated:
                    validation_status.append("✅ Email: Valid") 
                else:
                    email_attempts = EmailValidator.get_attempt_count(form_data['user_email'])
                    attempt_text = f" ({email_attempts} attempts)" if email_attempts > 0 else ""
                    validation_status.append(f"❌ Email: Needs validation{attempt_text}")
            
            if validation_status:
                status_text = " | ".join(validation_status)
                st.warning(f"⚠️ **Validation Status:** {status_text}")
    
    @staticmethod
    def _display_step1_success_popups(validation_results: Dict[str, Any]):
        
        # Model Combination Popup (only if matched from SOT)
        if validation_results['matched_models']:
            model1, model2 = validation_results['matched_models']
            
            # Get brand colors for visual display
            model1_config = ModelBrandManager.get_model_config(model1)
            model2_config = ModelBrandManager.get_model_config(model2)
            
            st.markdown(f"""
            <div style="background: #e3f2fd; 
                        padding: 0.5rem; border-radius: 10px; margin: 0.5rem 0; 
                        border-left: 4px solid {model1_config['color']};">
                <h4 style="margin: 0; color: #0d47a1;">⚔️ SxS Comparison</h4>
                <p style="margin: 0.1rem 0; font-size: 1.2rem; font-weight: 600;">
                    <span style="color: {model1_config['color']};">{model1}</span> 
                    <span style="color: #0d47a1;"> vs </span>
                    <span style="color: {model2_config['color']};">{model2}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Task ID & Metadata Popup
        qid_data = validation_results['question_id_data']
        col1, col2 = st.columns(2)
        
        if qid_data.get('task_id'):
                st.info(f"🔍 **Task ID:** {qid_data['task_id']}")
        
        with col1:
            if qid_data.get('project_type'):
                st.success(f"📂 **Project Type:** {qid_data['project_type']}")
        with col2:
            if qid_data.get('language'):
                st.success(f"🌐 **Language:** {qid_data['language']}")
    
    @staticmethod
    def render_image_upload():
        """Render image upload page."""
        st.header("2️⃣ Image Upload")
        
        if SecurityManager.is_workflow_completed():
            UIComponents.display_workflow_completion_status()
            return
        
        if not UIComponents.is_step_completed("Metadata Input"):
            st.error("⚠️ Prerequisites Missing: Please complete Step 1 (Metadata Input) first.")
            return
        
        col1, col2 = st.columns(2)

        # Display current setup
        with col1: st.markdown(f"""
        <div class="info-card">
            <h4>📋 Current Setup</h4>
            <p><strong>Comparison:</strong> {Utils.sanitize_html_output(st.session_state.model1)} vs {Utils.sanitize_html_output(st.session_state.model2)}</p>
            <p><strong>Question ID:</strong> {Utils.sanitize_html_output(st.session_state.question_id[:50])}...</p>
            <p><strong>Language:</strong> {Utils.sanitize_html_output(st.session_state.sot_language)}</p>
            <p><strong>Project Type:</strong> {Utils.sanitize_html_output(st.session_state.sot_project_type)}</p>
            <p><strong>Session ID:</strong> {Utils.sanitize_html_output(st.session_state.get('session_id', 'Unknown')[:8])}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col2.markdown(f"""
        <div class="info-card">
            <h4>ℹ️ Guidelines</h4>
            <p>– Include screenshots of <strong>both conversations</strong> (Gemini and its competitor).</p>
            <p>– <strong>Do not crop the screenshots</strong> — the full <strong>horizontal</strong> interface must be visible, including the model name.</p>
            <p>– Ensure <strong>all conversation turns</strong> are captured.</p>
            <p>– Place the screenshots in the correct <strong>chronological order</strong>.</p>
        </div>
        """, unsafe_allow_html=True)


        
        col1, col2 = st.columns(2)
        
        # Model 1 upload
        with col1:
            PageManager._render_model_upload_section(st.session_state.model1, "model1", "🔵")
        
        # Model 2 upload  
        with col2:
            PageManager._render_model_upload_section(st.session_state.model2, "model2", "🔴")
        
        # Save button
        PageManager._render_save_images_button()
        PageManager._show_next_step_button("Image Upload")
    
    @staticmethod
    def _render_model_upload_section(model_name: str, model_key: str, icon: str):
        """Render upload section for a model."""
        upload_locked = st.session_state.get('form_data_locked', False)
        upload_class = "upload-section locked" if upload_locked else "upload-section"
        
        st.markdown(f"""
        <div class="{upload_class}">
            <h3>{icon} {Utils.sanitize_html_output(model_name)} Screenshots</h3>
            <p>Upload interface screenshots and responses</p>
            {'<p><strong>🔒 LOCKED - Workflow completed</strong></p>' if upload_locked else ''}
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_images = st.file_uploader(
            f"Upload {model_name} screenshots",
            type=AppConfig.SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=True,
            key=f"{model_key}_images_upload",
            help=f"Upload screenshots of the {model_name} interface and responses",
            disabled=upload_locked
        )
        
        # Check if we have temp images stored (persists across reruns)
        temp_images_key = f"{model_key}_images_temp"
        existing_temp_images = st.session_state.get(temp_images_key, [])
        
        # Use uploaded images if available, otherwise use existing temp images
        images_to_process = uploaded_images if uploaded_images else existing_temp_images
        
        if images_to_process and not upload_locked:
            # Only validate file sizes if these are newly uploaded images
            if uploaded_images:
                valid_files = [img for img in uploaded_images if Utils.validate_file_size(img)]
                
                if len(valid_files) != len(uploaded_images):
                    st.error(f"Some files too large (max {AppConfig.MAX_FILE_SIZE_MB}MB each)")
                
                if valid_files:
                    # Store valid files as temp images
                    st.session_state[temp_images_key] = valid_files
                    images_to_process = valid_files
                else:
                    images_to_process = []
            
            if images_to_process:
                st.success(f"📁 {len(images_to_process)} valid image(s) uploaded for {model_name}")
                
                with st.expander("🔍 Preview & Reorder Images", expanded=True):
                    reordered_images = PageManager._create_reorderable_preview(images_to_process, model_name, model_key)
                    # Update temp images with reordered version
                    st.session_state[temp_images_key] = reordered_images
    
    @staticmethod
    def _create_reorderable_preview(images: List[BinaryIO], model_name: str, model_key: str) -> List[BinaryIO]:
        """Create reorderable image preview with persistent state."""
        if SecurityManager.is_workflow_completed():
            st.info("🔒 **Workflow completed** - Image reordering is locked.")
            for i, img in enumerate(images):
                st.image(img, caption=f"Position {i+1}: {model_name}", use_container_width=True)
            return images
        
        # Initialize reordered list with proper state management
        reorder_key = f"{model_key}_reordered"
        
        # Only initialize if we don't have existing reordered state or if images changed
        if reorder_key not in st.session_state or len(st.session_state[reorder_key]) != len(images):
            st.session_state[reorder_key] = list(images)
        
        current_images = st.session_state[reorder_key]
        
        # Check if order was modified from original
        original_order_different = current_images != list(images)
        if original_order_different and len(current_images) == len(images):
            st.success("✅ **Order modified** - Remember to click 'Save Images' to confirm changes")
        
        # Display images with controls
        for i, img in enumerate(current_images):
            col_img, col_controls = st.columns([5, 1])
            
            with col_img:
                try:
                    st.image(img, caption=f"Position {i+1}: {model_name}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying image {i+1}: {str(e)}")
            
            with col_controls:
                st.markdown(f"**#{i+1}**")
                
                # Move up button
                if i > 0:
                    move_up_key = f"{model_key}_up_{i}_{len(current_images)}"  # Add length to make key unique
                    if st.button("⬆️", help="Move up", key=move_up_key):
                        # Swap images
                        current_images[i], current_images[i-1] = current_images[i-1], current_images[i]
                        st.session_state[reorder_key] = current_images
                        st.rerun()
                
                # Move down button
                if i < len(current_images) - 1:
                    move_down_key = f"{model_key}_down_{i}_{len(current_images)}"  # Add length to make key unique
                    if st.button("⬇️", help="Move down", key=move_down_key):
                        # Swap images
                        current_images[i], current_images[i+1] = current_images[i+1], current_images[i]
                        st.session_state[reorder_key] = current_images
                        st.rerun()
            
            # Add separator between images (except for the last one)
            if i < len(current_images) - 1:
                st.markdown('<hr style="margin: 0.25rem 0; border: 0.5px solid #f0f0f0;">', unsafe_allow_html=True)
        
        return st.session_state[reorder_key]
    
    @staticmethod
    def _render_save_images_button():
        """Render save images button."""
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            model1_images = st.session_state.get('model1_images_temp', [])
            model2_images = st.session_state.get('model2_images_temp', [])
            
            save_disabled = (st.session_state.get('form_data_locked', False) or
                           not (model1_images and model2_images))
            
            if st.button("💾 Save Images", type="primary", use_container_width=True, disabled=save_disabled):
                if model1_images and model2_images:
                    # Get final reordered images
                    final_model1 = st.session_state.get("model1_reordered", model1_images)
                    final_model2 = st.session_state.get("model2_reordered", model2_images)
                    
                    # Save to main session state
                    st.session_state.model1_images = final_model1
                    st.session_state.model2_images = final_model2
                    
                    st.success("✅ Images saved with your chosen order! You can now proceed to Step 3.")
                    st.balloons()
                    
                    # Show confirmation
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.write(f"**{st.session_state.model1}:** {len(final_model1)} images")
                    with col_info2:
                        st.write(f"**{st.session_state.model2}:** {len(final_model2)} images")
                else:
                    st.error("❌ Please upload images for both models.")
    
    @staticmethod
    def render_pdf_generation():
        """Render PDF generation page."""
        st.header("3️⃣ PDF Generation")
        
        if SecurityManager.is_workflow_completed():
            UIComponents.display_workflow_completion_status()
            return
        
        required_keys = ['question_id', 'prompt_text', 'model1', 'model2', 'model1_images', 'model2_images']
        if not all(key in st.session_state for key in required_keys):
            st.error("⚠️ Prerequisites Missing: Please complete all previous steps.")
            return
        
        # Display summary
        st.subheader("📋 Final Review")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <h4>📝 Document Information</h4>
                <p><strong>Question ID:</strong> {Utils.sanitize_html_output(st.session_state.question_id[:50])}...</p>
                <p><strong>Model Comparison:</strong> {Utils.sanitize_html_output(st.session_state.model1)} vs {Utils.sanitize_html_output(st.session_state.model2)}</p>
                <p><strong>Prompt:</strong> {Utils.sanitize_html_output(st.session_state.prompt_text[:100])}...</p>
                <p><strong>Prompt Image:</strong> {"Yes" if st.session_state.get('prompt_image') else "No"}</p>
                <p><strong>Format:</strong> Google Slides 16:9 Widescreen</p>
                <p><strong>Session ID:</strong> {Utils.sanitize_html_output(st.session_state.get('session_id', 'Unknown')[:8])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-container">
                <div class="stat-card">
                    <h3>{len(st.session_state.model1_images)}</h3>
                    <p>{Utils.sanitize_html_output(st.session_state.model1)} Images</p>
                </div>
                <div class="stat-card">
                    <h3>{len(st.session_state.model2_images)}</h3>
                    <p>{Utils.sanitize_html_output(st.session_state.model2)} Images</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # PDF Generation
        PageManager._handle_pdf_generation()
        
        # PDF Preview and Download
        if st.session_state.get('pdf_generated') and 'pdf_buffer' in st.session_state:
            st.markdown("---")
            PageManager._render_pdf_preview_section()
        
        PageManager._show_next_step_button("PDF Generation")
    
    @staticmethod
    def _handle_pdf_generation():
        """Handle PDF generation logic."""
        pdf_already_generated = st.session_state.get('pdf_generated', False)
        
        if not pdf_already_generated:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔄 Generate PDF", type="primary", use_container_width=True):
                    with st.spinner("Generating PDF..."):
                        try:
                            with PDFGenerator() as pdf_gen:
                                pdf_buffer = pdf_gen.generate_pdf(
                                    st.session_state.question_id,
                                    st.session_state.prompt_text,
                                    st.session_state.model1,
                                    st.session_state.model2,
                                    st.session_state.model1_images,
                                    st.session_state.model2_images,
                                    st.session_state.get('prompt_image')
                                )
                                
                                st.session_state.pdf_buffer = pdf_buffer
                                st.session_state.pdf_generated = True
                                st.session_state.pdf_generation_time = datetime.now().isoformat()
                                
                                st.success("✅ PDF generated successfully!")
                                st.balloons()
                                
                        except Exception as e:
                            st.error("❌ Failed to generate PDF. Please try again.")
                            st.error(f"Error: {str(e)}")
        else:
            # Show regeneration options
            generation_time = st.session_state.get('pdf_generation_time', 'Unknown')
            st.success(f"✅ **PDF Already Generated** at {generation_time}")
            
            PageManager._render_regeneration_options()
    
    @staticmethod
    def _render_regeneration_options():
        """Render PDF regeneration options."""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.expander("🔄 Advanced: Regenerate PDF", expanded=False):
                st.warning("⚠️ **Warning**: Regenerating will replace the current PDF.")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔄 Regenerate PDF", type="secondary", use_container_width=True):
                        st.session_state.pdf_generated = False
                        if 'pdf_buffer' in st.session_state:
                            del st.session_state.pdf_buffer
                        st.success("🔄 Ready to regenerate PDF")
                        st.rerun()
                
                with col_b:
                    if st.button("🆕 Start New Session", type="primary", use_container_width=True):
                        SecurityManager.reset_session()
                        st.session_state.current_page = "Metadata Input"
                        st.success("🆕 New session started!")
                        st.rerun()
    
    @staticmethod
    def _render_pdf_preview_section():
        """Render PDF preview and download section."""
        st.subheader("📄 PDF Preview")
        
        # Create PDF preview
        try:
            st.session_state.pdf_buffer.seek(0)
            pdf_data = st.session_state.pdf_buffer.read()
            b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            
            pdf_display = f"""
            <div style="text-align: center;">
                <iframe src="data:application/pdf;base64,{b64_pdf}" 
                        width="100%" height="600px" 
                        style="border: none; border-radius: 5px;">
                </iframe>
            </div>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying PDF preview: {str(e)}")
        
        # Download section
        filename = Utils.generate_filename(st.session_state.model1, st.session_state.model2)
        
        st.info(f"🪪 **Filename:** {filename}")
        st.info(f"🏋️‍♀️ **File Size:** {len(pdf_data) / 1024:.1f} KB")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download PDF",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                type="secondary",
                use_container_width=True
            )
    
    @staticmethod
    def render_upload_to_drive():
        """Render upload to drive page with bulletproof button management."""
        st.header("4️⃣ Upload to Drive & Submit")
        
        if SecurityManager.is_workflow_completed():
            UIComponents.display_workflow_completion_status()
            PageManager._render_session_summary()
            PageManager._render_new_session_button()
            return
        
        if not st.session_state.get('pdf_generated'):
            st.error("⚠️ Prerequisites Missing: Please complete Step 3 (PDF Generation) first.")
            return
        
        # Ensure email was validated in Step 1
        if not st.session_state.get('email_validated') or not st.session_state.get('user_email'):
            st.error("⚠️ Email validation missing. Please return to Step 1 to validate your email.")
            return
        
        # Get PDF info
        filename = Utils.generate_filename(st.session_state.model1, st.session_state.model2)
        st.session_state.pdf_buffer.seek(0)
        pdf_data = st.session_state.pdf_buffer.read()
        file_size_kb = len(pdf_data) / 1024
        
        st.markdown("""
        <div style="background: #16213e; color: white; padding: 2rem; border-radius: 15px; margin: 1rem 0;">
            <h3 style="text-align: center; margin-bottom: 2rem;">📋 Final Submission</h3>
        """, unsafe_allow_html=True)
        
        # Display pre-validated email
        st.markdown("### ✅ Pre-validated Email")
        st.success(f"📧 **Email:** {st.session_state.user_email}")
        
        # Form data display
        PageManager._render_form_data_display(filename, file_size_kb)
        
        # === PHASE 1: CHECK IF WE SHOULD PROCESS ===
        # Check if drive upload should be processed
        if ButtonManager.should_process('drive_upload'):
            PageManager._process_drive_upload(filename)
        
        # Check if form submission should be processed
        if ButtonManager.should_process('form_submit'):
            PageManager._process_form_submission(filename, file_size_kb)
        
        # === PHASE 2: RENDER UI BASED ON CURRENT STATES ===
        # Drive upload section
        PageManager._render_drive_upload_section(filename)
        
        # Final submission section
        PageManager._render_final_submission_section(filename, file_size_kb)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def _process_drive_upload(filename: str):
        """Process drive upload - Phase 2 actual processing."""
        st.info("🚀 **Processing Drive Upload** - Please wait...")
        
        try:
            # Perform the actual upload
            upload_result = FormProcessor.process_drive_upload(st.session_state.user_email, filename)
            
            if upload_result.get('success'):
                # Success - complete the button processing
                ButtonManager.complete_processing('drive_upload', success=True)
                
                st.success("✅ Upload completed successfully!")
                st.success(f"🔗 **Drive URL:** {upload_result.get('drive_url', 'Generated')}")
                
                # Brief pause to show success before rerun
                time.sleep(1)
                st.rerun()
                
            else:
                # Failed - reset button for retry
                ButtonManager.complete_processing('drive_upload', success=False)
                
                error_msg = upload_result.get('error_message', 'Unknown error')
                st.error(f"❌ Upload failed: {error_msg}")
                
                # Brief pause before rerun
                time.sleep(2)
                st.rerun()
                
        except Exception as e:
            # Exception - reset button for retry
            ButtonManager.complete_processing('drive_upload', success=False)
            
            st.error(f"❌ Upload error: {str(e)}")
            
            # Brief pause before rerun
            time.sleep(2)
            st.rerun()
    
    @staticmethod
    def _process_form_submission(filename: str, file_size_kb: float):
        """Process form submission - Phase 2 actual processing."""
        st.info("🚀 **Processing Form Submission** - Please wait...")
        
        try:
            # Perform the actual submission
            submission_result = FormProcessor.process_form_submission(
                st.session_state.user_email, filename, file_size_kb
            )
            
            if submission_result.get('success'):
                # Success - complete the button processing and lock permanently
                ButtonManager.complete_processing('form_submit', success=True)
                
                # Mark as submitted to prevent duplicates
                st.session_state.form_submitted = True
                
                st.success("🎉 Form submitted successfully!")
                st.balloons()
                
                # Brief pause to show success before rerun
                time.sleep(1)
                st.rerun()
                
            else:
                # Failed - reset button for retry
                ButtonManager.complete_processing('form_submit', success=False)
                
                error_msg = submission_result.get('error_message', 'Unknown error')
                st.error(f"❌ Submission failed: {error_msg}")
                
                # Brief pause before rerun
                time.sleep(2)
                st.rerun()
                
        except Exception as e:
            # Exception - reset button for retry
            ButtonManager.complete_processing('form_submit', success=False)
            
            st.error(f"❌ Submission error: {str(e)}")
            
            # Brief pause before rerun
            time.sleep(2)
            st.rerun()
    
    @staticmethod
    def _render_form_data_display(filename: str, file_size_kb: float):
        """Render form data display section."""
        st.markdown("### 📋 Form Data Review")
        
        # Display all form data in a clean format
        form_data = [
            ("Question ID", st.session_state.question_id[:50] + "..."),
            ("Prompt Text", st.session_state.prompt_text[:100] + "..."),
            ("Model 1", f"{st.session_state.model1} ({len(st.session_state.model1_images)} images)"),
            ("Model 2", f"{st.session_state.model2} ({len(st.session_state.model2_images)} images)"),
            ("PDF File", f"{filename} ({file_size_kb:.1f} KB)"),
            ("Has Prompt Image", "Yes" if st.session_state.get('prompt_image') else "No"),
        ]
        
        for label, value in form_data:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{label}:**")
            with col2:
                st.markdown(f"`{value}`")
    
    @staticmethod
    def _render_drive_upload_section(filename: str):
        """Render drive upload section with bulletproof button management."""
        st.markdown("### 🔗 Google Drive Upload")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Show current drive URL status
            current_drive_url = st.session_state.get('drive_url', '')
            
            if current_drive_url:
                # Create a container for URL display with copy functionality
                url_col1, url_col2 = st.columns([4, 1])
                
                with url_col1:
                    st.text_input("Drive URL", value=current_drive_url, disabled=True, key="drive_url_display")
                
                with url_col2:
                    # Add clipboard copy button
                    if st.button("📋", help="Copy URL to clipboard", key="copy_drive_url"):
                        # Use st.code with copy functionality
                        st.success("✅ URL copied!")
                
                # Show the copyable URL in a code block for easy copying
                st.code(current_drive_url, language=None)
                st.markdown(f"🔗 **[Open in Google Drive]({current_drive_url})**")
                st.info("💡 **Copy the URL above to submit on CrC platform**")
                
            else:
                button_state = ButtonManager.get_button_state('drive_upload')
                if button_state == "processing":
                    st.text_input("Drive URL", value="⚙️ Upload in progress...", disabled=True)
                else:
                    st.text_input("Drive URL", value="Will be generated after upload", disabled=True)
        
        with col2:
            # Get button display info based on current state
            button_info = ButtonManager.get_button_display_info('drive_upload')
            can_upload = SecurityManager.can_upload_to_drive()
            
            # Render button with current state
            button_clicked = st.button(
                button_info['text'],
                type=button_info['type'],
                disabled=button_info['disabled'] or not can_upload,
                key="drive_upload_btn"
            )
            
            # Handle button click - IMMEDIATE STATE CHANGE
            if button_clicked and can_upload and ButtonManager.is_button_ready('drive_upload'):
                # IMMEDIATELY change state to processing and rerun
                if ButtonManager.start_processing('drive_upload'):
                    st.rerun()
            
            # Show additional status info for processing state
            if ButtonManager.is_button_processing('drive_upload'):
                st.info("🔒 Processing...")
    
    @staticmethod
    def _render_final_submission_section(filename: str, file_size_kb: float):
        """Render final submission section with bulletproof button management."""
        st.markdown("### 📤 Final Submission")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Get button display info based on current state
            button_info = ButtonManager.get_button_display_info('form_submit')
            can_submit = SecurityManager.can_submit_form()
            
            # Render button with current state
            button_clicked = st.button(
                button_info['text'],
                type=button_info['type'],
                use_container_width=True,
                disabled=button_info['disabled'] or not can_submit,
                key="form_submit_btn"
            )
            
            # Handle button click - IMMEDIATE STATE CHANGE
            if button_clicked and can_submit and ButtonManager.is_button_ready('form_submit'):
                # IMMEDIATELY change state to processing and rerun
                if ButtonManager.start_processing('form_submit'):
                    st.rerun()
            
            # Show additional status info for processing state
            if ButtonManager.is_button_processing('form_submit'):
                st.info("🔒 Processing...")
            
            # Show submission requirements if button is ready but can't submit
            if ButtonManager.is_button_ready('form_submit') and not can_submit:
                requirements = []
                if not st.session_state.get('email_validated', False):
                    requirements.append("❌ Email validation required")
                if not st.session_state.get('drive_url_generated', False):
                    requirements.append("❌ Drive upload required")
                if not st.session_state.get('drive_url', ''):
                    requirements.append("❌ Drive URL missing")
                
                if requirements:
                    st.warning("**Requirements:** " + " | ".join(requirements))
    
    @staticmethod
    def _render_session_summary():
        """Render final session summary."""
        st.markdown("---")
        st.subheader("📋 Final Session Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🔍 **Question ID:** {st.session_state.get('question_id', 'N/A')[:50]}...")
            st.info(f"🤖 **Models:** {st.session_state.get('model1', 'N/A')} vs {st.session_state.get('model2', 'N/A')}")
            st.info(f"📧 **Email:** {st.session_state.get('user_email', 'N/A')}")
        
        with col2:
            if st.session_state.get('drive_url'):
                st.info(f"🔗 **Drive URL:** [View File]({st.session_state.drive_url})")
            st.info(f"🆔 **Session ID:** {st.session_state.get('session_id', 'N/A')[:16]}")
            st.info(f"✅ **Status:** Form submitted successfully")
    
    @staticmethod
    def _render_new_session_button():
        """Render new session button."""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🆕 Start New Comparison", type="primary", use_container_width=True):
                SecurityManager.reset_session()
                st.session_state.current_page = "Metadata Input"
                st.success("🆕 Ready for a new comparison!")
                st.rerun()
    
    @staticmethod
    def render_help():
        """Render help page."""
        st.header("❓ Help & Documentation")
        
        if SecurityManager.is_workflow_completed():
            st.success("🔒 **Workflow Completed** - Your comparison has been successfully submitted.")
        
        tab1, tab2, tab3 = st.tabs(["📋 Instructions", "🔧 Troubleshooting", "📊 Examples"])
        
        with tab1:
            PageManager._render_help_instructions()
        
        with tab2:
            PageManager._render_help_troubleshooting()
        
        with tab3:
            PageManager._render_help_examples()
    
    @staticmethod
    def _render_help_instructions():
        """Render help instructions."""
        st.markdown("""
        ### 📋 How to Use This App
        
        #### 1️⃣ Metadata Input
        - Enter the **Question ID** (🛈 top right in CrC task)
        - Enter your **authorized email alias address** used in CrC
        - Enter the **Initial Prompt** used for both models
        - Optionally upload a **Prompt Image**
        - Both Question ID and email are validated automatically
        
        #### 2️⃣ Image Upload
        - Upload screenshots for both models
        - Preview images to ensure they're correct
        - **Reorder images** using up/down arrows if needed
        - Supports PNG, JPG, and JPEG formats
        
        #### 3️⃣ PDF Generation
        - Review your inputs in the summary
        - Click **Generate PDF** to create the document
        - **Preview** the PDF before downloading
        - **Download** the generated PDF file
        
        #### 4️⃣ Upload to Drive & Submit
        - Review all populated data from previous steps
        - Click **Upload** to upload to Drive (one-time only)
        - **Copy the Drive URL** using the clipboard button for easy pasting in CrC
        - Click **Submit** to complete the process (one-time only)
        
        ### 📄 PDF Structure
        1. **Title Page**: Question ID, Prompt, and optional image
        2. **First Model Brand Page**: Model name with brand styling
        3. **First Model Screenshots**: One image per slide
        4. **Second Model Brand Page**: Model name with brand styling
        5. **Second Model Screenshots**: One image per slide
        """)
    
    @staticmethod
    def _render_help_troubleshooting():
        """Render troubleshooting help."""
        st.markdown("""
        ### 🔧 Troubleshooting
        
        #### Common Issues:
        - **Question ID validation fails**: Ensure Question ID exists in SOT spreadsheet
        - **Email validation fails**: Ensure email is in authorized alias list
        - **No model combination found**: Question ID must have matching model pairing in SOT
        - **File too large**: All files must be under 50MB
        - **PDF generation fails**: Check image formats and try again
        - **Upload fails**: Ensure stable internet connection
        - **Form submission disabled**: Complete all required steps first
        - **"Already submitted" error**: Session is locked after completion - start new session
        - **Submit button disabled**: Complete Drive upload first
        
        #### Security-Related Issues:
        - **"Submission in progress"**: Wait for current submission to complete
        - **Multiple clicks not working**: System prevents duplicate submissions
        - **Session locked**: Start new session for additional comparisons
        
        #### Best Practices:
        - Use high-resolution screenshots (1920x1080 recommended)
        - Compress large images before upload using online tools
        - Ensure images are in supported formats (PNG, JPG, JPEG)
        - Complete all validations in Step 1 before proceeding
        - **Use the clipboard button (📋)** to copy Drive URL for CrC submission
        - **Do not refresh page during uploads** - may cause session issues
        - **Complete workflow in one session** - avoid leaving partially completed
        - **Click buttons only once** - system shows immediate feedback                    
        """)
    
    @staticmethod
    def _render_help_examples():
        """Render help examples."""
        st.markdown("""
        ### 📊 Examples
        
        #### Google SxS Model Recognition:
        This app automatically recognizes model combinations from tasks assigned in Google Crowd Compute (CrC):
        - **Question ID** → **Model pairing**
        - Example: `coach_P128631...` → `Bard 2.5 Pro vs. AIS 2.5 PRO`
        
        #### Sample Question ID:
        ```
        a5009505a2b411ff7b174326bb33306a+bard_data+coach_P128631_quality_sxs_e2e_experience_learning_and_academic_help_frozen_pool_human_eval_en-US-50+INTERNAL+en:18019373568084263285
        ```
        **Extracted Task ID:**
        ```
        coach_P128631_quality_sxs_e2e_experience_learning_and_academic_help_frozen_pool_human_eval_en-US-50
        ```
        **Auto-populated from SOT:**
        - Language: `en-US`
        - Model Comparison: `Bard 2.5 Pro vs. AIS 2.5 Pro`
        - Project Type: `Learning & Academic Help`
        
        #### Brand Recognition Examples:
        - **"Bard 2.5 Pro"** → Google Blue + "Gemini" brand
        - **"AIS 2.5 PRO"** → Google Blue + "Google AI Studio" brand  
        - **"cGPT o3"** → OpenAI Green + "OpenAI" brand
        - **"Claude"** → Anthropic Orange + "Anthropic" brand
        
        #### Sample Email Formats:
        ```
        ops-chiron-nonstem-en-us-007@invisible.co
        ops-chiron-coding-en-us-007@invisible.co
        ops-chiron-math-en-us-007@invisible.co
        your.name@invisible.email
        ```
        """)
    
    @staticmethod
    def _show_next_step_button(current_page: str):
        """Show next step button if current step is completed."""
        if SecurityManager.is_workflow_completed():
            return
        
        if not UIComponents.is_step_completed(current_page):
            return
        
        next_step_map = {
            "Metadata Input": "Image Upload",
            "Image Upload": "PDF Generation", 
            "PDF Generation": "Upload to Drive"
        }
        
        next_step = next_step_map.get(current_page)
        if not next_step:
            return
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            emoji_map = {"Image Upload": "2️⃣", "PDF Generation": "3️⃣", "Upload to Drive": "4️⃣"}
            button_text = f"Continue to {emoji_map.get(next_step, '▶️')} {next_step}"
            
            if st.button(button_text, type="primary", use_container_width=True, key=f"next_to_{next_step}"):
                st.session_state.current_page = next_step
                st.rerun()

# ============================================================================
# SIDEBAR COMPONENTS
# ============================================================================

class SidebarManager:
    """Manage sidebar components and navigation."""
    
    @staticmethod
    def render_navigation():
        """Render main navigation."""
        st.sidebar.title("🧭 Navigation")
        
        nav_options = [
            "1️⃣ Metadata Input",
            "2️⃣ Image Upload",
            "3️⃣ PDF Generation", 
            "4️⃣ Upload to Drive",
            "❓ Help"
        ]
        
        page_mapping = {
            "1️⃣ Metadata Input": "Metadata Input",
            "2️⃣ Image Upload": "Image Upload",
            "3️⃣ PDF Generation": "PDF Generation",
            "4️⃣ Upload to Drive": "Upload to Drive",
            "❓ Help": "Help"
        }
        
        # Filter options based on workflow completion
        if SecurityManager.is_workflow_completed():
            nav_options = ["4️⃣ Upload to Drive", "❓ Help"]
        
        # Find current selection
        current_nav_selection = None
        for nav_option, page_name in page_mapping.items():
            if page_name == st.session_state.current_page:
                current_nav_selection = nav_option
                break
        
        if current_nav_selection is None or current_nav_selection not in nav_options:
            current_nav_selection = nav_options[0]
        
        # Format options with status indicators
        def format_nav_option(x):
            page_name = page_mapping.get(x, x)
            if SecurityManager.is_workflow_completed() and x not in ["4️⃣ Upload to Drive", "❓ Help"]:
                return f"{x} 🔒"
            elif UIComponents.is_step_completed(page_name):
                return f"{x} ✅"
            else:
                return x
        
        selected_nav = st.sidebar.radio(
            "Choose Step:",
            nav_options,
            index=nav_options.index(current_nav_selection),
            format_func=format_nav_option
        )
        
        # Update current page with security checks
        target_page = page_mapping[selected_nav]
        
        if SecurityManager.is_navigation_allowed(target_page):
            st.session_state.previous_page = st.session_state.current_page
            st.session_state.current_page = target_page
        else:
            st.sidebar.error("🔒 Navigation to this page is restricted")
    
    @staticmethod
    def render_session_info():
        """Render current session information."""
        if 'question_id' in st.session_state:
            st.sidebar.markdown("### 📋 Current Session")
        
            
            if 'model1' in st.session_state and 'model2' in st.session_state:
                # Show with brand colors
                model1_config = ModelBrandManager.get_model_config(st.session_state.model1)
                model2_config = ModelBrandManager.get_model_config(st.session_state.model2)
                
                st.sidebar.markdown(f"""
                **Models:** 
                <span style="color: {model1_config['color']};">{st.session_state.model1}</span> vs 
                <span style="color: {model2_config['color']};">{st.session_state.model2}</span>
                """, unsafe_allow_html=True)
            
            st.sidebar.info(f"**Question ID:** {st.session_state.question_id[:20]}...")

            if st.session_state.get('sot_language'):
                st.sidebar.info(f"**Language:** {st.session_state.sot_language}")
            
            if st.session_state.get('sot_project_type'):
                st.sidebar.info(f"**Project:** {st.session_state.sot_project_type}")
            
            if st.session_state.get('user_email'):
                st.sidebar.info(f"**Email:** {st.session_state.user_email[:25]}...")
    
    @staticmethod
    def render_session_stats():
        """Render session statistics."""
        if any(key in st.session_state for key in ['model1_images', 'model2_images']):
            st.sidebar.markdown("### 📊 Session Stats")
            
            if 'model1_images' in st.session_state:
                st.sidebar.metric("Model 1 Images", len(st.session_state.model1_images))
            
            if 'model2_images' in st.session_state:
                st.sidebar.metric("Model 2 Images", len(st.session_state.model2_images))

# ============================================================================
# MAIN APPLICATION CONTROLLER
# ============================================================================

def main():
    """Main application controller."""
    # Initialize application
    st.set_page_config(**PAGE_CONFIG)
    SecurityManager.initialize_session_state()
    
    # Check for session expiration
    if SecurityManager.is_session_expired():
        st.error("🕒 Session expired. Please refresh the page to start a new session.")
        if st.button("🔄 Refresh Page"):
            SecurityManager.reset_session()
            st.rerun()
        return
    
    # Load custom CSS
    UIComponents.load_custom_css()
    
    # Display main header
    st.markdown("""
    <div class="main-header">
        <h1>🖨️ SxS Model Comparison PDF Generator</h1>
        <p>Generate standardized PDF documents for side-by-side LLM comparisons</p>
        <small style="opacity: 0.5;">Chiron EDU</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Handle workflow completion
    if SecurityManager.is_workflow_completed():
        UIComponents.display_workflow_completion_status()
        allowed_pages = ["Help", "Upload to Drive"]
        if st.session_state.current_page not in allowed_pages:
            st.session_state.current_page = "Upload to Drive"
    
    # Render sidebar components (simplified)
    SidebarManager.render_navigation()
    SidebarManager.render_session_info()
    SidebarManager.render_session_stats()
    
    # Display step indicator
    UIComponents.display_step_indicator(st.session_state.current_page)
    
    # Route to appropriate page
    page_handlers = {
        "Metadata Input": PageManager.render_metadata_input,
        "Image Upload": PageManager.render_image_upload,
        "PDF Generation": PageManager.render_pdf_generation,
        "Upload to Drive": PageManager.render_upload_to_drive,
        "Help": PageManager.render_help,
    }
    
    handler = page_handlers.get(st.session_state.current_page)
    if handler:
        handler()
    else:
        st.error(f"Unknown page: {st.session_state.current_page}")

if __name__ == "__main__":
    main()