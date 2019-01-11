----------------------------------------------------------------------------------
-- Author: Laurentiu-Cristian Duca
-- License: GNU GPL
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.float_sizes.all;

   
entity layer1 is
   port ( clk, rst : in std_logic;
          X: in inputs;
          thresholds_l1: in thresholds_l1_type;
			 Wt: in layer1_weights_t;
          R: out std_logic_vector(BNN_NEURONS_PER_HIDDEN_LAYER_N-1 downto 0));
end layer1;

architecture layer1_arch of layer1 is

component FPAdd_5_10_F50_uid2 
	  port ( clk, rst : in std_logic;
          X : in  std_logic_vector(5+10+2 downto 0);
          Y : in  std_logic_vector(5+10+2 downto 0);
          R : out  std_logic_vector(5+10+2 downto 0));
end component;

component compare_float
   port ( x : in  std_logic_vector(EXPONENT_SIZE+MANTISSA_SIZE+2 downto 0);
          y : in  std_logic_vector(EXPONENT_SIZE+MANTISSA_SIZE+2 downto 0);
          greater : out  std_logic;
          equal : inout std_logic);
end component;
			 
	--signal X00_concat_01, X01_concat_01, X02_concat_01, X03_concat_01: std_logic_vector(EXPONENT_SIZE+MANTISSA_SIZE+2 downto 0);
	signal X_concat_01: inputs_float_concat;
	signal thresholds_l1_concat_01: thresholds_l1_concat_type;
	signal Y0, Y1, Yo: thresholds_l1_concat_type;
	signal greater, equal: greater_type;
	
begin
	X_concat_i:
	for i in 0 to BNN_INPUTS_N-1 generate
	begin
		X_concat_j:
		for j in 0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1 generate  
		begin  
			X_concat_01(i,j) <= "01" & X(i) when (Wt(j)(i) = '1') else 
				"01" & (not X(i)(EXPONENT_SIZE+MANTISSA_SIZE)) & X(i)(EXPONENT_SIZE+MANTISSA_SIZE-1 downto 0);
		end generate;
	end generate;
	
	thresholds_l1_concat:
	for j in 0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1 generate  
	begin  
		thresholds_l1_concat_01(j) <= "01"&thresholds_l1(j);
	end generate;

	instances:
	for j in 0 to BNN_NEURONS_PER_HIDDEN_LAYER_N-1 generate  
	begin  
		n01: FPAdd_5_10_F50_uid2
				port map ( clk=>clk, rst=>rst, X=>X_concat_01(0,j), Y=>X_concat_01(1,j), R=>Y0(j));
		n23: FPAdd_5_10_F50_uid2
				port map ( clk=>clk, rst=>rst, X=>X_concat_01(2,j), Y=>X_concat_01(3,j), R=>Y1(j));   
		ny: FPAdd_5_10_F50_uid2
				port map ( clk=>clk, rst=>rst, X=>Y0(j), Y=>Y1(j), R=>Yo(j));	
		nc: compare_float
				port map ( x=>Yo(j), y=>thresholds_l1_concat_01(j), greater=>greater(j), equal=>equal(j));
		R(j) <= greater(j);
	end generate;

end layer1_arch;

