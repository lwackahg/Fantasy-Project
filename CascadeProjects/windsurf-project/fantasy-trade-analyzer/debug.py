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
        logger.propagate = False
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handlers = [
            h
            for h in list(logger.handlers)
            if isinstance(h, logging.FileHandler)
            and str(getattr(h, 'baseFilename', '')).lower().endswith('debug.log')
        ]
        if file_handlers:
            fh = file_handlers[0]
            for extra in file_handlers[1:]:
                logger.removeHandler(extra)
        else:
            fh = logging.FileHandler('debug.log')
            logger.addHandler(fh)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        
        # Console handler
        stream_handlers = [
            h
            for h in list(logger.handlers)
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        if stream_handlers:
            ch = stream_handlers[0]
            for extra in stream_handlers[1:]:
                logger.removeHandler(extra)
        else:
            ch = logging.StreamHandler()
            logger.addHandler(ch)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        
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
