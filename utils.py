import json
import nnsight
from nnsight import LanguageModel
from nnsight.intervention import InterventionProxy
from typing import List, Literal, Optional, Tuple, Union

def import_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def export_to_txt(output, filepath='output.txt'):
    with open(filepath, 'a') as f:
        f.write(output + "\n") 

def generate_tokens(model: LanguageModel, 
                    prompts: Union[List[str], str], 
                    n_tokens: int,
                    REMOTE: bool = False) -> List[str]:
    """Generate tokens from prompts.
    Args:
        model (LanguageModel): Language model.
        prompts (List[str]): List of prompts.
        n_tokens (int): Number of tokens to generate.
    Returns:
        Tensor: Tensor of shape (batch_size, seq_len) containing token ids.
    """
    model.tokenizer.pad_token = model.tokenizer.eos_token
    with model.generate(prompts, max_new_tokens = n_tokens) as tracer:
        for i in range(n_tokens):
            tracer.next()

        generated_tokens = model.generator.output.save()
    return generated_tokens # Tensor of shape (batch_size, seq_len) containing token ids

def my_viz(
    textArray: str,
    textValues: List[float],
    textHover: List[str],
    line_length: int,
    filename: Optional[str] = None,
):
    js_string = """function createHTML(textArray, textValues, textHover) {
    const container = document.createElement('div');
    container.style.position = 'relative';
    container.style.lineHeight = '30px';
    container.style.wordWrap = 'break-word';

    textArray.forEach((word, index) => {
        const span = document.createElement('span');
        span.textContent = word;
        span.style.padding = '5px';
        span.style.backgroundColor = interpolateColor(textValues[index]);
        span.style.cursor = 'pointer';
        span.style.position = 'relative';
        span.style.color = (textValues[index] >= 0.25 && textValues[index] <= 0.75) ? "black" : "white";

        const hoverDiv = document.createElement('div');
        hoverDiv.textContent = textHover[index];
        hoverDiv.style.width = '250px';
        hoverDiv.style.height = '100px';
        hoverDiv.style.position = 'absolute';
        hoverDiv.style.display = 'none';
        hoverDiv.style.justifyContent = 'center';
        hoverDiv.style.alignItems = 'center';
        hoverDiv.style.background = '#fff';
        hoverDiv.style.border = '1px solid black';
        hoverDiv.style.textAlign = 'center';
        hoverDiv.style.padding = '10px';
        hoverDiv.style.boxSizing = 'border-box';
        hoverDiv.style.top = '100%';
        hoverDiv.style.left = '50%';
        hoverDiv.style.color = 'black';
        hoverDiv.style.zIndex = '999';
        
        span.onmouseover = () => {
            hoverDiv.style.display = 'flex';
            const rect = span.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            if (rect.left - containerRect.left < 125) {
                hoverDiv.style.left = '0%';
                hoverDiv.style.transform = 'translateX(0%)';
            } else if (containerRect.right - rect.right < 125) {
                hoverDiv.style.left = '100%';
                hoverDiv.style.transform = 'translateX(-100%)';
            } else {
                hoverDiv.style.left = '50%';
                hoverDiv.style.transform = 'translateX(-50%)';
            }
        };


        span.onmouseout = () => { hoverDiv.style.display = 'none'; };

        span.appendChild(hoverDiv);
        container.appendChild(span);
    });

    return container;
}

function interpolateColor(value) {
    if (value < 0.5) {
        blue = 255;
        green = Math.round(255 * (2 * value));
        red = Math.round(255 * (2 * value));
    } else {
        blue = Math.round(255 * (2.0 - (2 * value)));
        green = Math.round(255 * (2.0 - (2 * value)));
        red = 255;
    }
    return `rgb(${red}, ${green}, ${blue})`;
}

// Usage: append the returned HTML object to your desired element
// document.body.appendChild(createHTML(["word1", "word2"], [0.1, 0.9], ["hover1", "hover2"]));
"""

    html_string = "<br>" * 5 + f"""<div id="my-viz"></div>
<script>
{js_string}
document.querySelector("#my-viz").appendChild(createHTML({textArray}, {textValues}, {textHover}));
</script>
""" + "<br>" * 10

    if filename is None:
        display(HTML(html_string))
    else:
        with open(filename, "w") as f:
            f.write(html_string)
        print(f"Saved at {filename!r}")


# For GPT2-Small
# def get_logit_diff_per_layer(model: LanguageModel, prompts: List[str], answer_token_ids: List[Int], per_prompt: bool = False) -> Float:
#     """Compute the difference between the logit of the two answer tokens.

#     Args:
#         logits (Tensor): Tensor of shape (batch_size, seq_len, vocab_size).
#         answer_tokens (List[Int]): List of answer tokens.

#     Returns:
#         Float: Difference between the logit of the two answer tokens.
#     """
#     with model.forward(remote=REMOTE) as runner:
#         with runner.invoke(prompts) as invoker:
#             attn_values = []
#             mlp_values = []
#             for layer in range(n_layers):
#                 attn_values.append(model.transformer.h[layer].attn.output[0][:, -1]) #[batch, d_model]
#                 mlp_values.append(model.transformer.h[layer].mlp.output[:, -1]) #[batch, d_model]
            
#             batch_size = len(prompts)
#             print("Batch size: ", batch_size, "Prompts:", prompts)

#             residual_final_pre_ln = model.transformer.h[-1].output[0][:, -1] #[batch, d_model]
#             print(f"{residual_final_pre_ln.shape=}")
#             residual_final_sf = residual_final_pre_ln.std(-1, keepdim=True) #[components, batch, 1]
#             print(f"{residual_final_sf.shape=}")

#             assert len(attn_values) == len(mlp_values) == n_layers, f"{len(attn_values)=} != {len(mlp_values)=} != {n_layers}"
#             assert attn_values[0].shape == mlp_values[0].shape == residual_final_pre_ln.shape ==(batch_size, d_model), f"{attn_values[0].shape=} != {mlp_values[0].shape=} != {residual_final_pre_ln.shape} != {(batch_size, d_model)=}  "

#             # Scale values by std of final residual
#             attn_values = t.stack(attn_values) / residual_final_sf #[components, batch, d_model], components = n_layers*2
#             print(f"{attn_values.shape=}")
#             mlp_values = t.stack(mlp_values) / residual_final_sf #[components, batch, d_model], components = n_layers*2
#             print(f"{mlp_values.shape=}")

#             # Calculate logit difference
#             attn_logits = model.lm_head(attn_values) #[components, batch, vocab_size]
#             print(f"{attn_logits.shape=}")
#             attn_logit_diff = (attn_logits[:, :, answer_token_ids[0]] - attn_logits[:, :, answer_token_ids[1]]).save() #[components, batch]
#             print(f"{attn_logit_diff.shape=}")
#             mlp_logits = model.lm_head(mlp_values) #[components, batch, vocab_size]
#             print(f"{mlp_logits.shape=}")
#             mlp_logit_diff = (mlp_logits[:, :, answer_token_ids[0]] - mlp_logits[:, :, answer_token_ids[1]]).save() #[components, batch]
#             print(f"{mlp_logit_diff.shape=}")

#     if per_prompt:
#         return attn_logit_diff.value, mlp_logit_diff.value #[components, batch]
#     else:
#         return attn_logit_diff.value.mean(-1), mlp_logit_diff.value.mean(-1) #[components,]
    

# def patch_act_residual_suffix(model: LanguageModel, 
#                        receiver_prompts: List[str], 
#                        source_prompts: List[str], 
#                        answer_token_ids: List[Int], 
#                        target_layers: List[int], 
#                        target_pos: int,
#                        normalizing_prompts: Optional[List[str]] = None,
#                        per_prompt: bool = False) -> Float:

#     if normalizing_prompts is not None:
#         tokens = model.tokenizer(receiver_prompts + source_prompts + normalizing_prompts, return_tensors='pt', padding=True)['input_ids'].to(device)
#         receiver_tokens = tokens[:len(receiver_prompts)]
#         source_tokens = tokens[len(receiver_prompts):len(receiver_prompts)+len(source_prompts)]
#         norm_tokens = tokens[-len(normalizing_prompts):]
#     else:
#         tokens = model.tokenizer(receiver_prompts + source_prompts, return_tensors='pt', padding=True)['input_ids'].to(device)
#         receiver_tokens = tokens[:len(receiver_prompts)]
#         source_tokens = tokens[len(receiver_prompts):]
#         norm_tokens = None

#     tokens = model.tokenizer(receiver_prompts + source_prompts, return_tensors='pt', padding=True)['input_ids'].to(device)
#     receiver_tokens = tokens[:len(receiver_prompts)]
#     source_tokens = tokens[len(receiver_prompts):]
#     print(f"{tokens.shape=}")
#     seq_len = tokens.shape[-1]
#     print(f"{seq_len=}")
#     n_layers = model.config.num_hidden_layers
    
#     # Get and store residual output for harmless prompts
#     assert len(receiver_prompts) == len(source_prompts)
    
#     with model.forward(remote=REMOTE) as runner:

#         # Store residual output per layer for SOURCE prompts
#         source_resid_dict = {}
#         with runner.invoke(source_tokens) as invoker:
#             for layer in range(n_layers):
#                 for pos in range(seq_len+target_pos, seq_len):
#                     assert seq_len == model.model.layers[layer].output[0].shape[1], f"{seq_len=} != {model.model.layers[layer].output[0].shape[1]=}, {model.model.layers[layer].output[0].shape=}"
#                     source_resid_dict[(layer, pos)] = model.model.layers[layer].output[0][:, pos].mean(0).save()  #[d_model] at selected pos averaged over batch
#             logits = model.lm_head.output[:, -1] #[batch, vocab_size]
#             source_refusal_score = (logits[:, answer_token_ids[0]] - logits[:, answer_token_ids[1]]).save() #[batch,]
        
#         # Get source/norm refusal score to compare against [SOURCE/NORM]
#         # if normalizing_prompts is not None:
#         #     print("Using normalizing prompts for source_refusal_score.")
#         #     with runner.invoke(norm_tokens) as invoker:
#         #         logits = model.lm_head.output[:, -1] #[batch, vocab_size]
#         #         source_refusal_score = (logits[:, answer_token_ids[0]] - logits[:, answer_token_ids[1]]).save() #[batch,]
        
#         # Get receiver refusal score to compare against [RECEIVER]
#         with runner.invoke(receiver_tokens) as invoker:
#             logits = model.lm_head.output[:, -1] #[batch, d_model] at -1 position
#             receiver_refusal_score = (logits[:, answer_token_ids[0]] - logits[:, answer_token_ids[1]]).save()

#         # Run forward pass on harmful prompts with intervention [RECEIVER]
#         intervened_refusal_score = {}
#         # for layer in range(n_layers):
#         for layer in target_layers:
#             for pos in range(seq_len+target_pos, seq_len):
#                 with runner.invoke(receiver_tokens) as invoker:
#                     model.model.layers[layer].output[0][:, pos] = source_resid_dict[(layer, pos)]
#                     logits = model.lm_head.output[:, -1]
#                     intervened_refusal_score[(layer, pos)] = (logits[:, answer_token_ids[0]] - logits[:, answer_token_ids[1]]).save() #[batch,]


#     # Get difference in residual score between intervention-harmless and harmful prompts (how much does patching in harmful restores refusal)
#     print(f"{intervened_refusal_score.keys()=}, \n {source_refusal_score.shape=}")
#     all_intervened_refusal_score = einops.rearrange(t.stack([score.value for score in intervened_refusal_score.values()]), '(layer pos) batch -> layer pos batch', layer = len(target_layers)) #n_layers)

#     #print(f"{all_intervened_refusal_score.shape=}, {all_intervened_refusal_score=} \n {source_refusal_score.value.shape=}, {source_refusal_score.value=} \n {receiver_refusal_score.value.shape=}, {receiver_refusal_score.value=}")
#     print(f"{(source_refusal_score.value - receiver_refusal_score.value).mean()=}")
#     print(f"{(all_intervened_refusal_score - receiver_refusal_score.value).mean(-1)=}")
#     refusal_score_diff = (all_intervened_refusal_score - receiver_refusal_score.value).mean(-1) / (source_refusal_score.value - receiver_refusal_score.value).mean() #[layer, pos, batch] -> [layer, pos]
#     # 1 = perfect intervention, 0 = no effect

#     return refusal_score_diff


# Activation patching - attn_output per layer
# def patch_attn_out_per_layer(model: LanguageModel, 
#                         receiver_prompts: List[str], 
#                         source_prompts: List[str], 
#                         answer_token_ids: List[Int], 
#                         target_layers: List[int], 
#                         target_pos: int,
#                         per_prompt: bool = False) -> Float:

#     assert len(receiver_prompts) == len(source_prompts) 
#     tokens = model.tokenizer(receiver_prompts + source_prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

#     # Get and store residual output for harmless prompts
#     with model.forward(remote=REMOTE) as runner:

#         # Run forward pass and store each attention head output 
#         attn_dict = {} # [d_head]
#         with runner.invoke(tokens[len(receiver_prompts):]) as invoker:
#             for layer in target_layers:   
#                 for pos in range(target_pos, 0):                 
#                     attn_dict[(layer, pos)] = model.model.layers[layer].self_attn.output[0][:, pos] #[batch seq d_model] -> [batch d_model]
#             logits = model.lm_head.output[:, -1] #[batch, vocab]
#             source_refusal_score = (logits[:, answer_token_ids[0]] - logits[:, answer_token_ids[1]]).mean().save() #[batch,] -> scalar (averaged over batch)
    
#         # Run forward pass on harmless prompts and store activation
#         with runner.invoke(tokens[:len(receiver_prompts)]) as invoker:
#             logits = model.lm_head.output[:, -1] #[batch, vocab] at -1 position
#             receiver_refusal_score = (logits[:, answer_token_ids[0]] - logits[:, answer_token_ids[1]]).mean().save() #[batch,] -> scalar (averaged over batch)

#         intervened_refusal_score = {}
#         for layer in target_layers:
#             for pos in range(target_pos, 0):
#                 with runner.invoke(tokens[:len(receiver_prompts)]) as invoker:
#                     model.model.layers[layer].self_attn.output[0][:, pos] = attn_dict[(layer, pos)]
#                     logits = model.lm_head.output[:, -1]
#                     intervened_refusal_score[(layer, pos)] = (logits[:, answer_token_ids[0]] - logits[:, answer_token_ids[1]]).save() #[batch,]

#     # Get residual score diff between intervention and harmful prompts (how much does patching in harmful restores refusal)
#     print(f"{intervened_refusal_score.keys()=}")
#     all_intervened_refusal_score = einops.rearrange(t.stack([score.value for score in intervened_refusal_score.values()]), '(layer pos) batch -> layer pos batch', layer = len(target_layers)).squeeze(1) #[layer, pos, batch]
#     refusal_score_diff_from_harmless = (all_intervened_refusal_score - receiver_refusal_score.value).mean(-1) / (source_refusal_score.value - receiver_refusal_score.value) #[layer, pos, batch] -> [layer, pos]
#     print(f"Source refusal score: {source_refusal_score.value.item(): .3f}, Receiver refusal score: {receiver_refusal_score.value.item(): .3f}")
#     print("Score:", refusal_score_diff_from_harmless)
#     return refusal_score_diff_from_harmless

# def compare_suffix_harmless_logit_diff(model: LanguageModel, 
#                           ld_unnormalized: Tensor,
#                           harmless_prompts: List[str], 
#                           harmful_prompts: List[str], 
#                           answer_token_ids: List[Int]) -> Float:
#     with model.forward(remote=REMOTE) as runner:
#         with runner.invoke(harmless_prompts) as invoker:
#                 logits = model.lm_head.output[:, -1] #[batch, d_model] at -1 position
#                 harmless_refusal_score = (logits[:, answer_token_ids[0]] - logits[:, answer_token_ids[1]]).mean().save()

#         with runner.invoke(harmful_prompts) as invoker:
#                 logits = model.lm_head.output[:, -1] #[batch, d_model] at -1 position
#                 harmless_refusal_score = (logits[:, answer_token_ids[0]] - logits[:, answer_token_ids[1]]).mean().save()

#     ld_norm = ld_unnormalized / (harmless_refusal_score.value - harmless_refusal_score.value)
#     return ld_norm


