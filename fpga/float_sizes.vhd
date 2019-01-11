library IEEE;
use IEEE.STD_LOGIC_1164.all;

package float_sizes is

function f_log2 (x : positive) return natural;

constant EXPONENT_SIZE: integer := 5;
constant MANTISSA_SIZE: integer := 10;
constant BNN_INPUTS_N: integer := 4;
constant BNN_HIDDEN_LAYERS_N: integer := 2;
constant BNN_NEURONS_PER_HIDDEN_LAYER_N: integer := 32;
-- some thresholds are greater than the number of neurons per layer.
constant BNN_LOG_NEURONS_PER_HIDDEN_LAYER_N: integer := (f_log2(BNN_NEURONS_PER_HIDDEN_LAYER_N)+1);
-- 
constant BNN_OUTPUTS_N: integer := 3;

-- HDL weights are the C transposed weights
type layer1_weights_t is array (0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1) of std_logic_vector(BNN_INPUTS_N-1 downto 0);
type layerh_weights_t is array (0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1) of std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
type layero_weights_t is array (0 to BNN_OUTPUTS_N-1) of std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
type inputs_float_concat is array(0 to BNN_INPUTS_N-1, 0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1) of std_logic_vector(EXPONENT_SIZE+MANTISSA_SIZE+2 downto 0);
type inputs is array (0 to BNN_INPUTS_N-1) of std_logic_vector(EXPONENT_SIZE+MANTISSA_SIZE downto 0);
type thresholds_l1_type is array (0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1) of std_logic_vector(EXPONENT_SIZE+MANTISSA_SIZE downto 0);
type thresholds_l1_concat_type is array (0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1) of std_logic_vector(EXPONENT_SIZE+MANTISSA_SIZE+2 downto 0);
type thresholds_lh_type is array (0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1) of std_logic_vector(BNN_LOG_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
type thresholds_lo_type is array (0 to BNN_OUTPUTS_N-1) of std_logic_vector(BNN_LOG_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
type greater_type is array (0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1) of std_logic;
end float_sizes;

package body float_sizes is
function f_log2 (x : positive) return natural is
      variable i : natural;
   begin
      i := 0;  
      while (2**i < x) and i < 31 loop
         i := i + 1;
      end loop;
      return i;
end function;
end float_sizes;