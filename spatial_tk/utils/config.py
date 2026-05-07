#!/usr/bin/env python3
"""
Configuration file utilities for spatial_tk.

This module provides functions to load and merge TOML configuration files
with command-line arguments.
"""

import argparse
import logging
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.9–3.10
    import tomli as tomllib  # type: ignore[import-not-found]
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a TOML configuration file.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        Dictionary containing the parsed TOML data
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the TOML file is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_file, 'rb') as f:
            config_dict = tomllib.load(f)
        return config_dict
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading config file {config_path}: {e}")


def convert_value(value: Any, target_type: type) -> Any:
    """
    Convert a value to the target type if possible.
    
    Handles common conversions:
    - Strings to int/float/bool
    - None values
    
    Args:
        value: Value to convert
        target_type: Target type to convert to
        
    Returns:
        Converted value
    """
    if value is None:
        return None
    
    # If already the right type, return as-is
    if isinstance(value, target_type):
        return value
    
    # Handle bool conversion
    if target_type == bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    # Handle int conversion
    if target_type == int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return int(float(value))  # Handle "1.0" -> 1
    
    # Handle float conversion
    if target_type == float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)
    
    # For other types, try direct conversion
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return value


def merge_config_with_args(
    command_name: str,
    config_dict: Dict[str, Any],
    args: argparse.Namespace,
    parser: Optional[argparse.ArgumentParser] = None
) -> argparse.Namespace:
    """
    Merge configuration file values with command-line arguments.
    
    CLI arguments take precedence over config values. Config values are applied
    only when the CLI argument wasn't explicitly provided (i.e., uses default).
    
    Args:
        command_name: Name of the command (e.g., 'concat', 'normalize')
        config_dict: Dictionary from load_config()
        args: Parsed argparse.Namespace from CLI
        parser: Optional ArgumentParser to determine defaults
        
    Returns:
        Modified argparse.Namespace with merged values
    """
    # Get command-specific section from config
    command_section = config_dict.get(command_name, {})
    
    if not command_section:
        logging.debug(f"No [{command_name}] section found in config file")
        return args
    
    args_dict = vars(args)
    
    # Get default values from parser if available
    # Extract defaults from parser actions
    defaults = {}
    if parser is not None:
        for action in parser._actions:
            if action.dest != 'help' and action.default is not argparse.SUPPRESS:
                defaults[action.dest] = action.default
    
    # Apply config values
    for config_key, config_value in command_section.items():
        # argparse stores arguments with underscores (converts hyphens)
        # Config keys use underscores, so match directly
        arg_key = None
        
        # Try exact match first (config uses underscores, argparse namespace uses underscores)
        if config_key in args_dict:
            arg_key = config_key
        # Try hyphen version (in case argparse stored it differently)
        elif config_key.replace('_', '-') in args_dict:
            arg_key = config_key.replace('_', '-')
        # Try finding by converting underscores/hyphens both ways
        else:
            for existing_key in args_dict.keys():
                # Normalize both keys by converting to underscores
                normalized_config = config_key.replace('-', '_')
                normalized_existing = existing_key.replace('-', '_')
                if normalized_config == normalized_existing:
                    arg_key = existing_key
                    break
        
        if arg_key is None:
            logging.debug(f"Skipping config key '{config_key}' (no matching CLI argument)")
            continue
        
        # Get current value and default value
        current_value = getattr(args, arg_key, None)
        default_value = defaults.get(arg_key)
        
        # Apply config value if current value matches default (i.e., wasn't explicitly set)
        # Or if current value is None/empty
        should_apply = False
        
        if current_value is None:
            should_apply = True
        elif default_value is not None and current_value == default_value:
            # Current value matches default, so CLI didn't override it
            should_apply = True
        elif isinstance(current_value, str) and current_value == '':
            should_apply = True
        
        if should_apply:
            # Convert config value to appropriate type
            # Infer type from config value or current value
            if isinstance(config_value, bool):
                target_type = bool
            elif isinstance(config_value, int):
                target_type = int
            elif isinstance(config_value, float):
                target_type = float
            elif current_value is not None:
                target_type = type(current_value)
            else:
                target_type = type(config_value)
            
            converted_value = convert_value(config_value, target_type)
            setattr(args, arg_key, converted_value)
            logging.debug(f"Applied config value: {arg_key} = {converted_value} (was {current_value})")
    
    return args

