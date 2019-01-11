--------------------------------------------------------------------------------
-- author: Laurentiu-Cristian Duca
--------------------------------------------------------------------------------
LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
 
ENTITY t1 IS
END t1;
 
ARCHITECTURE behavior OF t1 IS 
 
    -- Component Declaration for the Unit Under Test (UUT)
 
    COMPONENT FPAdd_5_10_F50_uid2
    PORT(
         clk : IN  std_logic;
         rst : IN  std_logic;
         X : IN  std_logic_vector(17 downto 0);
         Y : IN  std_logic_vector(17 downto 0);
         R : OUT  std_logic_vector(17 downto 0)
        );
    END COMPONENT;
    

   --Inputs
   signal clk : std_logic := '0';
   signal rst : std_logic := '0';
   signal X : std_logic_vector(17 downto 0) := (others => '0');
   signal Y : std_logic_vector(17 downto 0) := (others => '0');

 	--Outputs
   signal R : std_logic_vector(17 downto 0);

   -- Clock period definitions
   constant clk_period : time := 10 ns;

   -- aux variables
	signal normal_number : std_logic_vector (1 downto 0);
	signal xh, yh: std_logic_vector(15 downto 0);
	
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: FPAdd_5_10_F50_uid2 PORT MAP (
          clk => clk,
          rst => rst,
          X => X,
          Y => Y,
          R => R
        );

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
      -- hold reset state for 100 ns.
      wait for 100 ns;	

      wait for clk_period*10;

      -- insert stimulus here 
      wait;
   end process;
	
	normal_number <= b"01";
	xh <= x"42cc";
	yh <= x"3800";
	--X <= std_logic_vector(to_unsigned("01" & x"42cc", 18));
	X <= normal_number & xh;
	Y <= normal_number & yh;
END;
