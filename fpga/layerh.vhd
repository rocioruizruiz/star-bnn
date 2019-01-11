----------------------------------------------------------------------------------
-- author: Laurentiu-Cristian Duca
-- license: GNU GPL
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.std_logic_unsigned.all;
use work.float_sizes.all;

-- Uncomment the following library declaration if instantiating
-- any Xilinx primitives in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity layerh is
   port ( X: in std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
          thresholds_lh: in thresholds_lh_type;
			 Wt: in layerh_weights_t;
          R: out std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0));
end layerh;

architecture layerh_arch of layerh is

component population_counter
	port(
		WX: in std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
		popcnt: out std_logic_vector(BNN_LOG_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0));
end component;

signal WtX: layerh_weights_t;
signal popcnt: thresholds_lh_type;
begin

	WtXh_i:
	for i in 0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1 generate  
	begin
		WtX(i) <= not (Wt(i) xor X);
	end generate;

	popcounth_i:
	for i in 0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1 generate
	begin
		pc: population_counter port map (WX=>WtX(i), popcnt=>popcnt(i));
	end generate;

	Rh_i:
	for i in 0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1 generate  
	begin
		R(i) <= '1' when (popcnt(i) > thresholds_lh(i)) else '0';
	end generate;

end architecture;

