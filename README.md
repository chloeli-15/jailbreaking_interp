# Interpretability on Jailbreaking

## Activation patching

The patching metric has the following meaning:
- 1 = refusal behavior after patching is the same as on forward pass with harmful prompts
- 0 = refusal behavior after patching is the same as on forward pass with harmless prompts

I first tried patching the cumulative `attn_out` of each layer from the harmful to harmless pass. 

Observations:
- As expected, patching at the <obj> position (-7) restores refusal from layer ~5 onward. This can be thought of as replacing the harmless object ("cake") with a harmful object ("bomb"). 
- Similarly, patching at the -1 (":") position restores refusal from layer ~14-15 onward.
- More interestingly, patching at "." (-6) position almost restores refusal perfectly from layer 12 onward. This is surprising - it suggests that certain essential information for refusal is being stored here at this point in the model, which will then be moved to the -1 position. What might this information be?

