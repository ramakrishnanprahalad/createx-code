#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED control module for trash classification system.
Provides interfaces for controlling various types of LED indicators.
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LEDInterface(ABC):
    """Abstract base class for LED interfaces."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the LED."""
        pass
    
    @abstractmethod
    def set_color(self, color: Tuple[int, int, int]) -> None:
        """Set the LED color."""
        pass
    
    @abstractmethod
    def set_brightness(self, brightness: int) -> None:
        """Set the LED brightness."""
        pass
    
    @abstractmethod
    def turn_on(self) -> None:
        """Turn on the LED."""
        pass
    
    @abstractmethod
    def turn_off(self) -> None:
        """Turn off the LED."""
        pass
    
    @abstractmethod
    def flash(self, count: int, interval: float) -> None:
        """Flash the LED."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass


class GPIOLEDController(LEDInterface):
    """Implementation of LED interface using GPIO pins for Raspberry Pi."""
    
    def __init__(self, 
                 red_pin: int, 
                 green_pin: int, 
                 blue_pin: int,
                 use_pwm: bool = True,
                 pwm_freq: int = 100,
                 common_anode: bool = True) -> None:
        """
        Initialize GPIO LED controller.
        
        Args:
            red_pin: GPIO pin for red channel
            green_pin: GPIO pin for green channel
            blue_pin: GPIO pin for blue channel
            use_pwm: Whether to use PWM for brightness control
            pwm_freq: PWM frequency in Hz
            common_anode: Whether the LED is common anode (True) or common cathode (False)
        """
        self.red_pin = red_pin
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        self.use_pwm = use_pwm
        self.pwm_freq = pwm_freq
        self.common_anode = common_anode
        
        self.pins = [red_pin, green_pin, blue_pin]
        self.pwm_objects = []
        self.current_color = (0, 0, 0)
        self.brightness = 100  # 0-100
        self.is_on = False
        self.is_initialized = False
        self.flash_thread = None
        self.flash_stop_flag = False
    
    def initialize(self) -> bool:
        """Initialize GPIO pins and PWM objects."""
        try:
            # Import RPi.GPIO module
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
            except ImportError:
                logger.error("RPi.GPIO module not found. Is this running on a Raspberry Pi?")
                return False
            
            # Set GPIO mode
            self.GPIO.setmode(GPIO.BCM)
            
            # Setup pins
            for pin in self.pins:
                self.GPIO.setup(pin, GPIO.OUT)
            
            # Setup PWM if needed
            if self.use_pwm:
                for pin in self.pins:
                    pwm = self.GPIO.PWM(pin, self.pwm_freq)
                    pwm.start(0)
                    self.pwm_objects.append(pwm)
            
            self.is_initialized = True
            logger.info(f"GPIO LED initialized on pins {self.pins}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing GPIO LED: {str(e)}")
            return False
    
    def _convert_value(self, value: int) -> int:
        """
        Convert a 0-255 value to the appropriate value based on LED type.
        For common anode, invert the value (255 - value).
        Also scales to PWM duty cycle (0-100).
        """
        if self.use_pwm:
            # Scale from 0-255 to 0-100 for PWM duty cycle
            value = value * 100 // 255
        
        if self.common_anode:
            # For common anode, invert the value
            return 100 - value if self.use_pwm else 1 - (value > 0)
        else:
            return value
    
    def set_color(self, color: Tuple[int, int, int]) -> None:
        """
        Set the RGB color of the LED.
        
        Args:
            color: RGB color as (R, G, B) with values from 0-255
        """
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        self.current_color = color
        
        # Apply brightness
        adjusted_color = tuple(int(c * self.brightness / 100) for c in color)
        
        # Set the color
        if self.use_pwm:
            for pwm, value in zip(self.pwm_objects, adjusted_color):
                pwm.ChangeDutyCycle(self._convert_value(value))
        else:
            for pin, value in zip(self.pins, adjusted_color):
                self.GPIO.output(pin, self._convert_value(value) > 0)
        
        self.is_on = any(c > 0 for c in adjusted_color)
    
    def set_brightness(self, brightness: int) -> None:
        """
        Set the brightness of the LED.
        
        Args:
            brightness: Brightness level (0-100)
        """
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        self.brightness = max(0, min(100, brightness))
        
        # Update color with new brightness
        if self.is_on:
            self.set_color(self.current_color)
    
    def turn_on(self) -> None:
        """Turn on the LED with the current color and brightness."""
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        if not self.is_on:
            self.set_color(self.current_color)
    
    def turn_off(self) -> None:
        """Turn off the LED."""
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        self.set_color((0, 0, 0))
        self.is_on = False
    
    def flash(self, count: int, interval: float) -> None:
        """
        Flash the LED a specified number of times.
        
        Args:
            count: Number of flashes
            interval: Interval between on and off states in seconds
        """
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        # If already flashing, stop previous flashing
        if self.flash_thread and self.flash_thread.is_alive():
            self.flash_stop_flag = True
            self.flash_thread.join()
        
        # Reset stop flag
        self.flash_stop_flag = False
        
        # Start flashing in a separate thread
        self.flash_thread = threading.Thread(
            target=self._flash_task,
            args=(count, interval),
            daemon=True
        )
        self.flash_thread.start()
    
    def _flash_task(self, count: int, interval: float) -> None:
        """
        Flash task to be run in a separate thread.
        
        Args:
            count: Number of flashes
            interval: Interval between on and off states in seconds
        """
        original_color = self.current_color
        was_on = self.is_on
        
        try:
            for _ in range(count):
                if self.flash_stop_flag:
                    break
                
                self.turn_on()
                time.sleep(interval)
                
                if self.flash_stop_flag:
                    break
                
                self.turn_off()
                time.sleep(interval)
            
            # Restore original state
            if was_on:
                self.set_color(original_color)
            else:
                self.turn_off()
                
        except Exception as e:
            logger.error(f"Error in flash task: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        if not self.is_initialized:
            return
        
        # Stop flashing if active
        if self.flash_thread and self.flash_thread.is_alive():
            self.flash_stop_flag = True
            self.flash_thread.join(timeout=1.0)
        
        # Stop PWM if used
        if self.use_pwm:
            for pwm in self.pwm_objects:
                pwm.stop()
        
        # Clean up GPIO
        self.GPIO.cleanup(self.pins)
        self.is_initialized = False
        logger.info("GPIO LED resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()


class GpioZeroLEDController(LEDInterface):
    """Implementation of LED interface using gpiozero library for Raspberry Pi."""
    
    def __init__(self, 
                 red_pin: int, 
                 green_pin: int, 
                 blue_pin: int,
                 common_anode: bool = True) -> None:
        """
        Initialize gpiozero LED controller.
        
        Args:
            red_pin: GPIO pin for red channel
            green_pin: GPIO pin for green channel
            blue_pin: GPIO pin for blue channel
            common_anode: Whether the LED is common anode (True) or common cathode (False)
        """
        self.red_pin = red_pin
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        self.common_anode = common_anode
        
        self.pins = [red_pin, green_pin, blue_pin]
        self.led = None
        self.current_color = (0, 0, 0)
        self.brightness = 1.0  # 0-1
        self.is_on = False
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize gpiozero LED objects."""
        try:
            # Import gpiozero module
            try:
                from gpiozero import RGBLED
                self.RGBLED = RGBLED
            except ImportError:
                logger.error("gpiozero module not found. Is this running on a Raspberry Pi?")
                return False
            
            # Setup LED
            self.led = self.RGBLED(
                red=self.red_pin,
                green=self.green_pin,
                blue=self.blue_pin,
                active_high=not self.common_anode,  # Invert for common anode
                pwm=True
            )
            
            self.is_initialized = True
            logger.info(f"gpiozero LED initialized on pins {self.pins}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing gpiozero LED: {str(e)}")
            return False
    
    def set_color(self, color: Tuple[int, int, int]) -> None:
        """
        Set the RGB color of the LED.
        
        Args:
            color: RGB color as (R, G, B) with values from 0-255
        """
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        # Convert from 0-255 to 0-1
        r, g, b = [c / 255 for c in color]
        
        # Apply brightness
        r *= self.brightness
        g *= self.brightness
        b *= self.brightness
        
        # Set the color
        self.led.color = (r, g, b)
        self.current_color = color
        self.is_on = any(c > 0 for c in color)
    
    def set_brightness(self, brightness: int) -> None:
        """
        Set the brightness of the LED.
        
        Args:
            brightness: Brightness level (0-100)
        """
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        # Convert from 0-100 to 0-1
        self.brightness = max(0, min(100, brightness)) / 100
        
        # Update color with new brightness
        if self.is_on:
            self.set_color(self.current_color)
    
    def turn_on(self) -> None:
        """Turn on the LED with the current color and brightness."""
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        if not self.is_on and any(c > 0 for c in self.current_color):
            self.set_color(self.current_color)
        elif not self.is_on:
            # If no color set, use white
            self.set_color((255, 255, 255))
    
    def turn_off(self) -> None:
        """Turn off the LED."""
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        self.led.off()
        self.is_on = False
    
    def flash(self, count: int, interval: float) -> None:
        """
        Flash the LED a specified number of times.
        
        Args:
            count: Number of flashes
            interval: Interval between on and off states in seconds
        """
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        # Use gpiozero's built-in blink method
        r, g, b = [c / 255 for c in self.current_color]
        r *= self.brightness
        g *= self.brightness
        b *= self.brightness
        
        self.led.blink(
            on_time=interval,
            off_time=interval,
            n=count,
            background=True,
            on_color=(r, g, b)
        )
    
    def cleanup(self) -> None:
        """Clean up gpiozero resources."""
        if not self.is_initialized:
            return
        
        if self.led:
            self.led.close()
        
        self.is_initialized = False
        logger.info("gpiozero LED resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()


class DummyLEDController(LEDInterface):
    """Dummy LED controller for testing without actual hardware."""
    
    def __init__(self) -> None:
        """Initialize dummy LED controller."""
        self.current_color = (0, 0, 0)
        self.brightness = 100
        self.is_on = False
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize dummy LED."""
        self.is_initialized = True
        logger.info("Dummy LED initialized")
        return True
    
    def set_color(self, color: Tuple[int, int, int]) -> None:
        """Set the dummy LED color."""
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        self.current_color = color
        adjusted_color = tuple(int(c * self.brightness / 100) for c in color)
        
        logger.info(f"Dummy LED color set to {adjusted_color}")
        self.is_on = any(c > 0 for c in adjusted_color)
    
    def set_brightness(self, brightness: int) -> None:
        """Set the dummy LED brightness."""
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        self.brightness = max(0, min(100, brightness))
        logger.info(f"Dummy LED brightness set to {self.brightness}%")
        
        if self.is_on:
            adjusted_color = tuple(int(c * self.brightness / 100) for c in self.current_color)
            logger.info(f"Dummy LED color adjusted to {adjusted_color}")
    
    def turn_on(self) -> None:
        """Turn on the dummy LED."""
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        if not self.is_on:
            self.is_on = True
            logger.info(f"Dummy LED turned on with color {self.current_color}")
    
    def turn_off(self) -> None:
        """Turn off the dummy LED."""
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        self.is_on = False
        logger.info("Dummy LED turned off")
    
    def flash(self, count: int, interval: float) -> None:
        """Flash the dummy LED."""
        if not self.is_initialized:
            logger.error("LED not initialized. Call initialize() first.")
            return
        
        logger.info(f"Dummy LED flashing {count} times with {interval}s interval")
        
        # Simulate flashing
        for i in range(count):
            logger.info(f"Dummy LED flash {i+1}/{count}: ON")
            time.sleep(interval)
            logger.info(f"Dummy LED flash {i+1}/{count}: OFF")
            time.sleep(interval)
    
    def cleanup(self) -> None:
        """Clean up dummy LED resources."""
        if self.is_initialized:
            logger.info("Dummy LED resources cleaned up")
            self.is_initialized = False


class LEDFactory:
    """Factory class for creating LED controllers."""
    
    @staticmethod
    def create_led(led_type: str, **kwargs) -> LEDInterface:
        """
        Create an LED controller based on the given type.
        
        Args:
            led_type: Type of LED controller ('gpio', 'gpiozero', 'dummy')
            **kwargs: Additional parameters for the specific LED controller
            
        Returns:
            An LEDInterface implementation
            
        Raises:
            ValueError: If the LED type is not supported
        """
        if led_type.lower() == 'gpio':
            return GPIOLEDController(**kwargs)
        elif led_type.lower() == 'gpiozero':
            return GpioZeroLEDController(**kwargs)
        elif led_type.lower() == 'dummy':
            return DummyLEDController()
        else:
            raise ValueError(f"Unsupported LED type: {led_type}")


class LEDManager:
    """Manages multiple LEDs for trash classification feedback."""
    
    # Color codes for trash categories
    TRASH_CATEGORY_COLORS = {
        'recyclable_plastic': (0, 0, 255),    # Blue
        'recyclable_glass': (0, 255, 255),    # Cyan
        'recyclable_metal': (128, 128, 128),  # Gray
        'recyclable_paper': (255, 255, 0),    # Yellow
        'compostable': (0, 255, 0),           # Green
        'landfill': (255, 0, 0),              # Red
        'ewaste': (255, 0, 255),              # Magenta
        'unknown': (255, 255, 255),           # White
    }
    
    def __init__(self, leds: Dict[str, LEDInterface] = None) -> None:
        """
        Initialize LED manager.
        
        Args:
            leds: Dictionary of LEDs with names as keys
        """
        self.leds = leds or {}
        self.initialized = False
    
    def add_led(self, name: str, led: LEDInterface) -> None:
        """
        Add an LED to the manager.
        
        Args:
            name: Name for this LED
            led: LED controller
        """
        self.leds[name] = led
    
    def initialize(self) -> bool:
        """Initialize all LEDs."""
        success = True
        
        for name, led in self.leds.items():
            if not led.initialize():
                logger.error(f"Failed to initialize LED '{name}'")
                success = False
        
        self.initialized = success
        return success
    
    def set_all_color(self, color: Tuple[int, int, int]) -> None:
        """
        Set all LEDs to the same color.
        
        Args:
            color: RGB color as (R, G, B) with values from 0-255
        """
        if not self.initialized:
            logger.error("LED manager not initialized. Call initialize() first.")
            return
        
        for led in self.leds.values():
            led.set_color(color)
    
    def set_category_color(self, category: str) -> None:
        """
        Set all LEDs to the color corresponding to the trash category.
        
        Args:
            category: Trash category name
        """
        if not self.initialized:
            logger.error("LED manager not initialized. Call initialize() first.")
            return
        
        color = self.TRASH_CATEGORY_COLORS.get(category, self.TRASH_CATEGORY_COLORS['unknown'])
        self.set_all_color(color)
    
    def flash_category(self, category: str, count: int = 3, interval: float = 0.2) -> None:
        """
        Flash all LEDs with the color corresponding to the trash category.
        
        Args:
            category: Trash category name
            count: Number of flashes
            interval: Interval between on and off states in seconds
        """
        if not self.initialized:
            logger.error("LED manager not initialized. Call initialize() first.")
            return
        
        color = self.TRASH_CATEGORY_COLORS.get(category, self.TRASH_CATEGORY_COLORS['unknown'])
        
        for led in self.leds.values():
            led.set_color(color)
            led.flash(count, interval)
    
    def turn_all_off(self) -> None:
        """Turn off all LEDs."""
        if not self.initialized:
            logger.error("LED manager not initialized. Call initialize() first.")
            return
        
        for led in self.leds.values():
            led.turn_off()
    
    def cleanup(self) -> None:
        """Clean up all LED resources."""
        for name, led in self.leds.items():
            try:
                led.cleanup()
                logger.info(f"Cleaned up LED '{name}'")
            except Exception as e:
                logger.error(f"Error cleaning up LED '{name}': {str(e)}")
        
        self.initialized = False
    
    def __enter__(self):
        """Context manager enter method."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.cleanup()


# Example usage
if __name__ == "__main__":
    # Use dummy LED for testing
    led = LEDFactory.create_led('dummy')
    
    # Create LED manager
    manager = LEDManager()
    manager.add_led('main', led)
    
    try:
        # Initialize LEDs
        if manager.initialize():
            print("LEDs initialized successfully")
            
            # Test all colors
            for category, color in LEDManager.TRASH_CATEGORY_COLORS.items():
                print(f"Testing category: {category}")
                manager.set_category_color(category)
                time.sleep(1)
            
            # Flash for recyclable
            print("Flashing for recyclable_plastic")
            manager.flash_category('recyclable_plastic')
            time.sleep(3)
            
            # Turn off all LEDs
            manager.turn_all_off()
            print("All LEDs turned off")
            
        else:
            print("Failed to initialize LEDs")
    
    finally:
        # Clean up LED resources
        manager.cleanup()
        print("LED resources cleaned up") 