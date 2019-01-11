----------------------------------------------------------------------------------
-- author: Laurentiu-Cristian Duca
-- license: GNU GPL
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.std_logic_arith.all;
use IEEE.std_logic_unsigned.all;
use work.float_sizes.all;

entity inference is
   port ( clk, rst : in std_logic;
          partial_results: out thresholds_lo_type;
			 R: out std_logic_vector(BNN_OUTPUTS_N-1 downto 0));
end inference;

architecture inference_arch of inference is
signal X: inputs;
signal thresholds_l1: thresholds_l1_type;
signal thresholds_lh2: thresholds_lh_type;
signal thresholds_lo: thresholds_lo_type;
signal Wt_l1: layer1_weights_t;
signal Wt_lh2: layerh_weights_t;
signal Wt_lo: layero_weights_t;
signal Rl1: std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
signal Rlh2: std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0);
begin

li: entity work.layer1(layer1_arch)
   port map ( clk=>clk, rst=>rst, X=>X, thresholds_l1=>thresholds_l1, Wt=>Wt_l1, R=>Rl1);
lh2: entity work.layerh(layerh_arch)
   port map (X=>Rl1, thresholds_lh=>thresholds_lh2, Wt=>Wt_lh2, R=>Rlh2);
lo: entity work.layero(layero_arch)
   port map (X=>Rlh2, Wt=>Wt_lo, thresholds_lo=>thresholds_lo, partial_results=>partial_results, R=>R);

   process(clk, rst)
   begin
		if(rst='1') then
			-- last
			X(0) <= x"38f6";
			X(1) <= x"b440";
			X(2) <= x"2dab";
			X(3) <= x"b95d";
			-- i=2
			--X(0) <= x"34a8";
			--X(1) <= x"b144";
			--X(2) <= x"b88e";
			--X(3) <= x"bafc";
			thresholds_l1(0) <= x"b56a";
			thresholds_l1(1) <= x"3c18";
			thresholds_l1(2) <= x"b839";
			thresholds_l1(3) <= x"32d0";
			thresholds_l1(4) <= x"b356";
			thresholds_l1(5) <= x"bab5";
			thresholds_l1(6) <= x"3bbc";
			thresholds_l1(7) <= x"b1e8";
			--W(0)<="11011010"; -- bnn.c prints in reverse order: 01011011
			--W(1)<="01011001"; -- bnn.c prints in reverse order: 10011010
			--W(2)<="10010110"; -- bnn.c prints in reverse order: 01101001
			--W(3)<="10110100"; -- bnn.c prints in reverse order: 00101101
			Wt_l1(0) <= "0010";
			Wt_l1(1) <= "0101";
			Wt_l1(2) <= "1100";
			Wt_l1(3) <= "0011";
			Wt_l1(4) <= "1111";
			Wt_l1(5) <= "1000";
			Wt_l1(6) <= "0011";
			Wt_l1(7) <= "1101";

			thresholds_lh2(0) <= conv_std_logic_vector(4,4);
			thresholds_lh2(1) <= conv_std_logic_vector(3,4);
			thresholds_lh2(2) <= conv_std_logic_vector(4,4);
			thresholds_lh2(3) <= conv_std_logic_vector(4,4);
			thresholds_lh2(4) <= conv_std_logic_vector(4,4);
			thresholds_lh2(5) <= conv_std_logic_vector(4,4);
			thresholds_lh2(6) <= conv_std_logic_vector(3,4);
			thresholds_lh2(7) <= conv_std_logic_vector(3,4);
			Wt_lh2(0)<="01001001";
			Wt_lh2(1)<="10111111";
			Wt_lh2(2)<="10010100";
			Wt_lh2(3)<="10101011";
			Wt_lh2(4)<="10010100";
			Wt_lh2(5)<="01101101";
			Wt_lh2(6)<="10010110";
			Wt_lh2(7)<="10100110";
			
			thresholds_lo(0) <= conv_std_logic_vector(5,4);
			thresholds_lo(1) <= conv_std_logic_vector(9,4);
			thresholds_lo(2) <= conv_std_logic_vector(6,4);
			Wt_lo(0)<="00101001";
			Wt_lo(1)<="11011010";
			Wt_lo(2)<="11010110";

         elsif rising_edge(clk) then
         end if;
   end process;

end architecture;

