----------------------------------------------------------------------------------
-- author: Laurentiu-Cristian Duca
-- license: GNU GPL
-- compares half-float
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.float_sizes.all;

entity compare_float is
   port ( x : in  std_logic_vector(EXPONENT_SIZE+MANTISSA_SIZE+2 downto 0);
          y : in  std_logic_vector(EXPONENT_SIZE+MANTISSA_SIZE+2 downto 0);
          greater : out  std_logic;
          equal : inout std_logic);
end entity;

architecture compare_float_arch of compare_float is

signal xmsbits, ymsbits	: std_logic_vector (1 downto 0);
signal xsign, ysign: std_logic;
signal xexp, yexp: std_logic_vector(EXPONENT_SIZE-1 downto 0);
signal xmantissa, ymantissa: std_logic_vector(MANTISSA_SIZE-1 downto 0);

begin
-- assert x and y are not NaN or Inf
xmsbits <= x(EXPONENT_SIZE+MANTISSA_SIZE+2 downto EXPONENT_SIZE+MANTISSA_SIZE+1);
ymsbits <= y(EXPONENT_SIZE+MANTISSA_SIZE+2 downto EXPONENT_SIZE+MANTISSA_SIZE+1);
xsign <= x(EXPONENT_SIZE+MANTISSA_SIZE);
ysign <= y(EXPONENT_SIZE+MANTISSA_SIZE);
xexp <= x(EXPONENT_SIZE+MANTISSA_SIZE-1 downto MANTISSA_SIZE);
yexp <= y(EXPONENT_SIZE+MANTISSA_SIZE-1 downto MANTISSA_SIZE);
xmantissa <= x(MANTISSA_SIZE-1 downto 0);
ymantissa <= y(MANTISSA_SIZE-1 downto 0);

equal <= '1' when ((x = y) or ((xmsbits = ymsbits) and (xmsbits /= "01")))
	else '0';

greater <= '1' when ((equal /= '1') and
	(((xmsbits = "00") and (ysign = '1')) or 
	((xmsbits = "10") and (xsign = '0')) or
	((xsign = '0') and (ysign = '1')) or
	((xsign = '0') and (ysign = '0') and ((xexp > yexp) or ((xexp = yexp) and (xmantissa > ymantissa)))) or
	((xsign = '1') and (ysign = '1') and ((xexp < yexp) or ((xexp = yexp) and (xmantissa < ymantissa))))))
	else '0';

end compare_float_arch;

