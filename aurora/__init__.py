"""AURORA: Autonomous Root Cause Analysis for Microservices"""
__version__ = "1.0.1"
__author__ = "Jaykumar Maheshkar"

def __getattr__(name):
    if name == "get_settings":
        from aurora.config.settings import get_settings
        return get_settings
    raise AttributeError(name)
