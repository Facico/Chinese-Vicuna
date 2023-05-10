
import transformers
from utils import printf
import copy

class prompt:
    def __init__(self, tokenizer, max_len, add_eos=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_eos=add_eos

class instruct_prompt(prompt):
    prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    prompt_input = (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:{instruction}\n\n### Input:{input}\n\n### Response:"
    )
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def preprocess_gen(self, data_point):
        if 'history' not in data_point:
        # single instruction format {'instruction':..,'input':..}
            if 'input' in data_point:
                user_prompt = self.prompt_input.format_map(data_point)
            else:
                user_prompt = self.prompt.format_map(data_point)
        else:
        # multi turn format {'history':[..], 'input':[..]}
            user_prompt = "\n".join(["User:" + i['input']+"\n"+"Assistant:" + i['output'] for i in data_point['history']]) + "\nUser:" + data_point['input'] + "\nAssistant:"
            user_prompt = user_prompt[-maxlen:]
        user_prompt=self.prompt.format_map({'instruction':user_prompt})
        input_ids = self.tokenizer(user_prompt)["input_ids"]
        return input_ids

    def preprocess_train(self, data_point):
        # single instruction format {'instruction':..,'input':..,'output':..}
        if 'instruction' in data_point:
            if 'input' in data_point:
                user_prompt = self.prompt_input.format_map(data_point)
            else:
                user_prompt = self.prompt.format_map(data_point)
            output = data_point["output"]
        # multi turn format {'input':[..], 'output':[..]}
        else:
            user_prompt = ''
            lens = len(data_point['input'])
            for i in range(lens-1):
                user_prompt += self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
            user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1]})
            user_prompt = self.prompt.format_map({'instruction': user_prompt})
            output = data_point['output'][-1]

        len_user_prompt_tokens = (len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len + 1,
        )["input_ids"])- 1)  # no eos token
        full_tokens = self.tokenizer(
            user_prompt + output,
            truncation=True,
            max_length=self.max_len + 1,
            padding="max_length",
        )["input_ids"][:-1]
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def data_collator(self,):
        return transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def postprocess(self, text, render=True):
        #import pdb;pdb.set_trace()
        printf(text)
        output = text.split("### Response:")[1].strip()
        output = output.replace("Belle", "Vicuna")
        printf(output)
        if '###' in output:
            output = output.split("###")[0]
        if 'User' in output:
            output = output.split("User")[0]
        output = output.replace('�','').replace('</s>', '') 
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

class chat_prompt(prompt):
    prompt_pre = (
        "The following is a conversation between an AI assistant called Assistant and a human user called User. "
        "The assistant is intelligent, knowledgeable and polite to answer questions of user.\n\n"
    )
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def preprocess_gen(self, data_point):
        user_prompt = self.prompt_pre
        len_avail = self.max_len - len(self.tokenizer(user_prompt, add_special_tokens=False)['input_ids'])
        input_prompt = self.prompt_post.format_map({'input':data_point['input']})
        len_avail -= len(self.tokenizer(input_prompt, add_special_tokens=False)['input_ids'])
        lens = len(data_point['history'])
        tokenized_lens = []
        for i in range(lens):
            tmp_prompt = self.prompt_history.format_map(data_point['history'][i])
            tokenized_lens.append(len(self.tokenizer(tmp_prompt,add_special_tokens=False)["input_ids"]))
        
        # 启发式：/2 优先除前面的
        i = 0
        while sum(tokenized_lens) > len_avail and i < lens:
            history = data_point['history'][i]
            tmp_len1 = len(history['input'])
            tmp_len2 = len(history['output'])
            if tmp_len2 > tmp_len1:
                history['output'] = history['output'][:tmp_len2//2]
            else:
                history['input'] = history['input'][:tmp_len1//2]
            prompt = self.prompt_history.format_map(history)
            single_len =(len(self.tokenizer(prompt,add_special_tokens=False)["input_ids"]))
            tokenized_lens[i] = single_len
            i += 1
        total_len = sum(tokenized_lens)
        # 还不够的话 直接截断
        while total_len > len_avail and i < lens - 1 :
            total_len -= tokenized_lens[i]
            data_point['history'] = data_point['history'][1:]
            i += 1
        # 最终合并
        for i in range(lens):
            user_prompt += self.prompt_history.format_map(data_point['history'][i])
        user_prompt += input_prompt
        printf({'real_input:':user_prompt})
        inputs = self.tokenizer(user_prompt)["input_ids"]
        return inputs

    def preprocess_train(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})

        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) - 1 # remove extra eos
        if self.add_eos:
            full_tokens = self.tokenizer(
                user_prompt + data_point["output"][-1].strip(),
                truncation=True,
                padding=False,
                max_length=self.max_len,
            )["input_ids"] # need eos
        else:
            full_tokens = self.tokenizer(
                user_prompt + data_point["output"][-1].strip(),
                truncation=True,
                padding=False,
                max_length=self.max_len+1,
            )["input_ids"][:-1] # delete eos
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def data_collator(self,):
        return transformers.DataCollatorForSeq2Seq(self.tokenizer)

    def postprocess(self, text, render=False):
        output = text.split("Assistant:")[-1].strip()
        if 'User:' in output:
            output = output.split("User:")[0]
        output = output.replace('�','') 
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

    def get_data_collator():
        return transformers.DataCollatorForLanguageModeling