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

entity inference32 is
   port ( clk, rst : in std_logic;
          partial_results: out thresholds_lo_type;
			 R: out std_logic_vector(BNN_OUTPUTS_N-1 downto 0));
end inference32;

architecture inference32_arch of inference32 is
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

Wt_l1(0)<="0000";
Wt_l1(1)<="1010";
Wt_l1(2)<="0101";
Wt_l1(3)<="1001";
Wt_l1(4)<="0011";
Wt_l1(5)<="1100";
Wt_l1(6)<="1110";
Wt_l1(7)<="0010";
Wt_l1(8)<="0001";
Wt_l1(9)<="1010";
Wt_l1(10)<="1110";
Wt_l1(11)<="0101";
Wt_l1(12)<="1100";
Wt_l1(13)<="1010";
Wt_l1(14)<="1011";
Wt_l1(15)<="0000";
Wt_l1(16)<="0101";
Wt_l1(17)<="0101";
Wt_l1(18)<="1110";
Wt_l1(19)<="1101";
Wt_l1(20)<="1001";
Wt_l1(21)<="1101";
Wt_l1(22)<="1111";
Wt_l1(23)<="1110";
Wt_l1(24)<="0010";
Wt_l1(25)<="0010";
Wt_l1(26)<="1100";
Wt_l1(27)<="0101";
Wt_l1(28)<="0100";
Wt_l1(29)<="0011";
Wt_l1(30)<="0101";
Wt_l1(31)<="0101";
Wt_lh2(0)<="00100011000110001010001011011011";
Wt_lh2(1)<="11011100111011110001110111100101";
Wt_lh2(2)<="11011101111011110101110011110110";
Wt_lh2(3)<="00010000000110101010001110101011";
Wt_lh2(4)<="00100011010101001110001111010010";
Wt_lh2(5)<="11011100101111100001100001101100";
Wt_lh2(6)<="00100111100000010111011110011111";
Wt_lh2(7)<="11111100111011010001011001100110";
Wt_lh2(8)<="00100011110100000110010110010010";
Wt_lh2(9)<="00100001000110101100101110011111";
Wt_lh2(10)<="11111000000110111000100110101101";
Wt_lh2(11)<="10100000000100001010101110111011";
Wt_lh2(12)<="00001011000110101110001100001011";
Wt_lh2(13)<="00000011010101001100010010011010";
Wt_lh2(14)<="00100011000101011100001110011011";
Wt_lh2(15)<="11011110111001110001110001100100";
Wt_lh2(16)<="00100111110001010011011001010010";
Wt_lh2(17)<="00000110111011110001010001100110";
Wt_lh2(18)<="11001100011110110001110101100101";
Wt_lh2(19)<="11011101111111110001110001100100";
Wt_lh2(20)<="00110011000100101100101110101111";
Wt_lh2(21)<="00100001000101000110011010011011";
Wt_lh2(22)<="00100011010101001110011111010010";
Wt_lh2(23)<="11011100111011110001110001101100";
Wt_lh2(24)<="11011100010011111001100000100101";
Wt_lh2(25)<="00100011100111001111001111011011";
Wt_lh2(26)<="01000010000100011010101110111011";
Wt_lh2(27)<="11010100111011110001100000100101";
Wt_lh2(28)<="00100011110101000110000011011010";
Wt_lh2(29)<="11011000000010111001110101100100";
Wt_lh2(30)<="00100001100100000110001110110011";
Wt_lh2(31)<="11011010000100011100000110001111";
Wt_lo(0)<="11010110011000010111101101010001";
Wt_lo(1)<="10101101100111000001111000101111";
Wt_lo(2)<="00100000100011111110000010100110";

thresholds_l1(0) <= x"3c06"; thresholds_l1(1) <= x"ba29"; thresholds_l1(2) <= x"3ad3"; thresholds_l1(3) <= x"385d"; thresholds_l1(4) <= x"34f0"; thresholds_l1(5) <= x"b9b3"; thresholds_l1(6) <= x"b727"; thresholds_l1(7) <= x"b230"; thresholds_l1(8) <= x"3aa8"; thresholds_l1(9) <= x"bbde"; thresholds_l1(10) <= x"bc13"; thresholds_l1(11) <= x"3a7b"; thresholds_l1(12) <= x"b835"; thresholds_l1(13) <= x"baed"; thresholds_l1(14) <= x"21bc"; thresholds_l1(15) <= x"2c06"; thresholds_l1(16) <= x"35bf"; thresholds_l1(17) <= x"3b0d"; thresholds_l1(18) <= x"bca8"; thresholds_l1(19) <= x"3564"; thresholds_l1(20) <= x"3c00"; thresholds_l1(21) <= x"94fa"; thresholds_l1(22) <= x"33ec"; thresholds_l1(23) <= x"bb7e"; thresholds_l1(24) <= x"3543"; thresholds_l1(25) <= x"36e9"; thresholds_l1(26) <= x"b937"; thresholds_l1(27) <= x"b240"; thresholds_l1(28) <= x"2eaf"; thresholds_l1(29) <= x"3715"; thresholds_l1(30) <= x"b87f"; thresholds_l1(31) <= x"b1aa"; 
thresholds_lh2(0) <= conv_std_logic_vector(15,6); thresholds_lh2(1) <= conv_std_logic_vector(13,6); thresholds_lh2(2) <= conv_std_logic_vector(17,6); thresholds_lh2(3) <= conv_std_logic_vector(17,6); thresholds_lh2(4) <= conv_std_logic_vector(15,6); thresholds_lh2(5) <= conv_std_logic_vector(15,6); thresholds_lh2(6) <= conv_std_logic_vector(16,6); thresholds_lh2(7) <= conv_std_logic_vector(18,6); thresholds_lh2(8) <= conv_std_logic_vector(13,6); thresholds_lh2(9) <= conv_std_logic_vector(13,6); thresholds_lh2(10) <= conv_std_logic_vector(22,6); thresholds_lh2(11) <= conv_std_logic_vector(14,6); thresholds_lh2(12) <= conv_std_logic_vector(14,6); thresholds_lh2(13) <= conv_std_logic_vector(15,6); thresholds_lh2(14) <= conv_std_logic_vector(19,6); thresholds_lh2(15) <= conv_std_logic_vector(20,6); thresholds_lh2(16) <= conv_std_logic_vector(9,6); thresholds_lh2(17) <= conv_std_logic_vector(16,6); thresholds_lh2(18) <= conv_std_logic_vector(16,6); thresholds_lh2(19) <= conv_std_logic_vector(14,6); thresholds_lh2(20) <= conv_std_logic_vector(15,6); thresholds_lh2(21) <= conv_std_logic_vector(17,6); thresholds_lh2(22) <= conv_std_logic_vector(15,6); thresholds_lh2(23) <= conv_std_logic_vector(16,6); thresholds_lh2(24) <= conv_std_logic_vector(16,6); thresholds_lh2(25) <= conv_std_logic_vector(17,6); thresholds_lh2(26) <= conv_std_logic_vector(14,6); thresholds_lh2(27) <= conv_std_logic_vector(13,6); thresholds_lh2(28) <= conv_std_logic_vector(14,6); thresholds_lh2(29) <= conv_std_logic_vector(17,6); thresholds_lh2(30) <= conv_std_logic_vector(18,6); thresholds_lh2(31) <= conv_std_logic_vector(16,6); 
thresholds_lo(0) <= conv_std_logic_vector(19,6); thresholds_lo(1) <= conv_std_logic_vector(27,6); thresholds_lo(2) <= conv_std_logic_vector(22,6); 
         end if;
   end process;

end architecture;

