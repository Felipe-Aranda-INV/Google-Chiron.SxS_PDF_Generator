# 🖨️ SxS PDF Generator

A Streamlit web app that generates standardized PDF documents for side-by-side LLM model comparisons, with automated Google Sheets integration.
• Designed for AIT Inv agents working on Google’s Chiron EDU project for SxS human evaluations.

## Architecture
graph TD
    A[User Upload] --> B[Streamlit App]
    B --> C[PDF Generator]
    C --> D[Google Drive]
    B --> E[Apps Script API]
    E --> F[Google Sheets]
    
    style A fill:#e1f5fe
    style B fill:#4285f4,color:#fff
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#e8f5e8

## Features

• **Secure Workflow**: Anti-exploitation protection with session locking
• **Professional PDFs**: Google Slides 16:9 format with company branding
• **Automated Integration**: Direct upload to Google Drive and Sheets logging
• **Image Reordering**: Drag-and-drop interface for screenshot organization
• **Validation**: Authorized user and LLM combo verification with fallback support

## Quick Start

• **Configure**: Set webhook_url in Streamlit secrets
• **Deploy**: Upload to Streamlit Cloud or run locally
• **Access**: Navigate through the 4-step guided workflow

## Tech Stack

• **Frontend**: Streamlit with custom CSS styling
• **PDF Engine**: ReportLab with PIL image processing (future OCR impl.)
• **Integration**: Google Apps Script webhook API
• **Security**: Session-based locks and input validation


Production v2 • Chiron EDU • FA
