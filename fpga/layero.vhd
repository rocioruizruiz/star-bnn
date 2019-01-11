----------------------------------------------------------------------------------
-- author: Laurentiu-Cristian Duca
-- license: GNU GPL
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.std_logic_unsigned.all;
--use IEEE.std_logic_arith.all;

use work.float_sizes.all;

entity layero is
   port ( X: in std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
			 Wt: in layero_weights_t;
			 thresholds_lo: in thresholds_lo_type;
          partial_results: out thresholds_lo_type;
			 R: out std_logic_vector(BNN_OUTPUTS_N-1 downto 0));
end layero;

architecture layero_arch of layero is

component population_counter
	port(
		WX: in std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
		popcnt: out std_logic_vector(BNN_LOG_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0));
end component;

signal popcnt: thresholds_lo_type;
signal WtX: layero_weights_t;
--signal maximum: std_logic_vector(BNN_LOG_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
signal index_maximum: std_logic_vector(BNN_OUTPUTS_N-1 downto 0);
begin

	WtXo_i:
	for i in 0 to BNN_OUTPUTS_N-1 generate  
	begin
		WtX(i) <= not (Wt(i) xor X);
	end generate;


	popcounth_i:
	for i in 0 to BNN_OUTPUTS_N-1 generate
	begin
		pc: population_counter port map (WX=>WtX(i), popcnt=>popcnt(i));
	end generate;

	results_i:
	for i in 0 to BNN_OUTPUTS_N-1 generate
	begin
		partial_results(i) <= popcnt(i);
		--R(i) <= '1' when (popcnt(i) > thresholds_lo(i)) else '0';
		--R(i) <= '1' when (popcnt(i) = maximum) else '0';
		R(i) <= '1' when (std_logic_vector(to_unsigned(i, BNN_OUTPUTS_N)) = index_maximum) else '0';
	end generate;

	process(popcnt)
	variable max: std_logic_vector(BNN_LOG_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0) := (others => '0');
	variable index_max: std_logic_vector(BNN_OUTPUTS_N-1 downto 0) := (others => '0');
	begin
		for i in 0 to BNN_OUTPUTS_N-1 loop
			if(unsigned(max) < unsigned(popcnt(i))) then
				max := popcnt(i);
				--index_max := conv_std_logic_vector(i, BNN_OUTPUTS_N);
				index_max := std_logic_vector(to_unsigned(i, BNN_OUTPUTS_N));
			end if;
		end loop;
		--maximum <= max;
		index_maximum <= index_max;
	end process;
end architecture;

