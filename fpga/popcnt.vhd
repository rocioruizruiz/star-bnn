----------------------------------------------------------------------------------
-- author: Laurentiu-Cristian Duca
-- license: GNU GPL
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.std_logic_unsigned.all;
use work.float_sizes.all;

entity population_counter is
	port(
		WX: in std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
		popcnt: out std_logic_vector(BNN_LOG_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0));
end population_counter;

architecture population_counter_arch of population_counter is
begin
		process(WX)
		variable vpopcnt: std_logic_vector(BNN_LOG_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
		begin
			vpopcnt := (others => '0');
			l_popcnt : for k in 0 to (BNN_NEURONS_PER_HIDDEN_LAYER_N-1) loop
				vpopcnt := vpopcnt + WX(k);
			end loop l_popcnt;
			popcnt <= vpopcnt;
		end process;

end population_counter_arch;

