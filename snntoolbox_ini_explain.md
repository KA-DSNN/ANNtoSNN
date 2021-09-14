# SNN toolbox config file parameter explaining 

## cell
### reset
* possible values
  * a text with "subtraction" string in it or not.


* changes
  * ```python 
    if 'subtraction' in config.get('cell', 'reset'):
        self.v_reset = 'v = v - v_thresh'
    else:
        self.v_reset = 'v = v_reset'
    ```