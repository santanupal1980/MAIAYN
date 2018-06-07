<B>The Transformer model in Multisource Attention Is All You Need (MAIAYN)</B>
  
<b>The code and style was borrowed from the repository repository https://github.com/Lsdefine/attention-is-all-you-need-keras </b>


The current implementation is a concatenation of two separate self-attended encoders deliver to another self attended joint encoder.


<b> How to Run </b>
  
  1. you need to concatenate src1, src2 and tgt seprated by a TAB (\t).
  
  e.g., 
  
  $ paste trn_src1_file trn_src2_file trn_tgt_file > train.ms
  
  $ paste dev_src1_file dev_src2_file dev_tgt_file > dev.ms
  
  2. Update config.py
  
  3. run python trainMS.py
  
  4. python translateMS.py
