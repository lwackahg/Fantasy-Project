"""Debug module for Fantasy Trade Analyzer.

This module provides debug functionality while maintaining Claude desktop integration.
"""
import logging
from datetime import datetime
import streamlit as st
from contextlib import contextmanager

class DebugManager:
    def __init__(self):
        self.debug_mode = False
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('FantasyTradeAnalyzer')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler('debug.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def toggle_debug(self):
        """Toggle debug mode on/off."""
        self.debug_mode = not self.debug_mode
        self.log(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
        
    def log(self, message, level='info'):
        """Log a message with the specified level."""
        if not self.debug_mode and level == 'debug':
            return
            
        log_func = getattr(self.logger, level)
        log_func(message)
        
        try:
            # Only attempt UI logging if in a Streamlit context
            if self.debug_mode:
                with st.expander("Debug Log", expanded=True):
                    st.text(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
        except:
            # Silently pass if we're not in a Streamlit context
            pass

# Global debug manager instance
debug_manager = DebugManager()
