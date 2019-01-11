--------------------------------------------------------------------------------
-- test bench
--------------------------------------------------------------------------------
LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
use work.float_sizes.all;
 
ENTITY ti IS
END ti;
 
ARCHITECTURE behavior OF ti IS 
 
    -- Component Declaration for the Unit Under Test (UUT)
 
    COMPONENT inference
    PORT(
         clk : IN  std_logic;
         rst : IN  std_logic;
         partial_results: out thresholds_lo_type;
         R: out std_logic_vector(BNN_OUTPUTS_N-1 downto 0));
    END COMPONENT;
    

   --Inputs
   signal clk : std_logic := '0';
   signal rst : std_logic := '0';

 	--Outputs
   signal partial_results : thresholds_lo_type;
   signal R: std_logic_vector(BNN_OUTPUTS_N-1 downto 0);
   -- Clock period definitions
   constant clk_period : time := 10 ns;
 
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: inference PORT MAP (
          clk => clk,
          rst => rst,
			 partial_results=>partial_results,
			 R=>R);
			 
   -- Clock process definitions
   clk_process :process
   begin
		clk <= '0';
		wait for clk_period/2;
		clk <= '1';
		wait for clk_period/2;
   end process;
 

   -- Stimulus process
   stim_proc: process
   begin
		rst <= '1';
      -- hold reset state for 100 ns.
      wait for 100 ns;	
		rst <= '0';
		
      wait for clk_period*10;

      -- insert stimulus here 

      wait;
   end process;

END;
