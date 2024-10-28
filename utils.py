import re
from torch.utils.data import Dataset, DataLoader
import transformers
import copy
import torch
from datasets import load_dataset
import numpy as np

ANSER_INDICATOR = "The answer is "
gsm8k_opencompass_chat = [
    dict(role='user', content="Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step"),
    dict(role='assistant', content='Answer:\nAngelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4.\n'),
    dict(role='user', content="Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step"),
    dict(role='assistant', content="Answer:\nMark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201.\n"),
    dict(role='user', content="Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step"),
    dict(role='assistant', content="Answer:\nWhen Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140.\n"),
    dict(role='user', content="Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step"),
    dict(role='assistant', content='Answer:\nFor the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146.\n')
]
gsm8k_opencompass_prompt = [
    obj["content"]
    for obj in gsm8k_opencompass_chat
]
gsm8k_opencompass_prompt = "\n".join(gsm8k_opencompass_prompt)

gsm8k_cot_prompts = [
    "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.",
    "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.",
    "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    "A: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."
]
gsm8k_cot_prompts = "\n".join(gsm8k_cot_prompts)
AQuA_cot_prompts = [
    "Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? Answer Choices are: A)50; B)45; C)65; D)78; E)64.",
    "A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. The answer is (A).",
    "Q: If a / b = 3/4 and 8a + 5b = 22,then find the value of a. Answer Choices are: A)1/2; B)3/2; C)5/2; D)4/2; E)7/2.",
    "A: If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2. The answer is (B).",
    "Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer Choices are: A)53 km; B)55 km; C)52 km; D)60 km; E)50 km.",
    "A: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. The answer is (E).",
]
AQuA_cot_prompts = "\n".join(AQuA_cot_prompts)
OBQA_cot_prompts = [
    "Q: Poison causes harm to which of the following? Answer Choices are: A)a Tree; B)a robot; C)a house; D)a car.",
    "A: Poison will harm living things, only a tree is a living thing. The answer is (A).",
    "Q: As you look deeper into a Marbel you can see? Answer Choices are: A)the future; B)minut defects; C)colors; D)the other side.",
    "A: Marbel is not transparent, so you can not see the other side. Marbel does not necessarily have multiple colors. You will see minut defects. The answer is (B)",
    "Q: When food is reduced in the stomach? Answer Choices are: A)the mind needs time to digest; B)take a second to digest what I said; C)nutrients are being deconstructed; D)reader’s digest is a body of works.",
    "A: The food is being deconstructed in the stomach during digestion. The answer is (C)."
]
OBQA_cot_prompts = "\n".join(OBQA_cot_prompts)
ANLI_cot_prompts = [
    'Q: "Conceptually cream skimming has two basic dimensions - product and geography." Based on this premise, can we conclude the hypothesis "Product and geography are what make cream skimming work." is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.',
    'A: Based on "cream skimming has two basic dimensions" we can not infer that these two dimensions are what make cream skimming work. It is not possible to tell. The answer is (C)',
    'Q: "One of our member will carry out your instructions minutely." Based on this premise, can we conclude the hypothesis "A member of my team will execute your orders with immense precision." is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.',
    'A: "one of" means the same as "a member of", "carry out" means the same as "execute", and "minutely" means the same as "immense precision". So we can say yes. The answer is (A)',
    'Q: "Fun for adults and children." Based on this premise, can we conclude the hypothesis "Fun for only children." is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.',
    'A: "adults and children" contradicts "only children". So we can not conclude the hypothesis. The answer is (B).'
]
ANLI_cot_prompts = "\n".join(ANLI_cot_prompts)
self_weight_propmts = [
    "Below is a question and a candidate answer.",
    "Evaluate whether or not the answer is a good example.",
    "A good answer should be complete, clear, and comprehensive",
    "The answer sentence should be well organized without missing or irrelevant information.",
    "Use a number between 0 and 10 to represent the rating of the candidate answer.",
    "10 means the best and 0 means the worst.",
    "Please follow the format 'Score: <rating>'.\n Here are the question and candidate answer: \n"
]
self_weight_propmts = " ".join(self_weight_propmts)
StrategyQA_cot_prompts = [
    "Q: Do hamsters provide food for any animals? Answer Choices are: A)yes; B)no.",
    "A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. The answer is (A).",
    "Q: Could Brooke Shields succeed at University of Pennsylvania? Answer Choices are: A)yes; B)no.",
    "A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. The answer is (A).",
    "Q: Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls? Answer Choices are: A)yes; B)no.",
    "A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. The answer is (B)."
]
StrategyQA_cot_prompts = "\n".join(StrategyQA_cot_prompts)
DATASET_NAME_REGISTER = {
    "gsm8k",
    "aqua_rat",
    "ucinlp/drop",
    "ChilleD/SVAMP",
    "allenai/openbookqa",
    "facebook/anli",
    "ChilleD/StrategyQA"
}
DATASET_SPLIT = {
    "gsm8k": 7099,
    "ChilleD/SVAMP": 665,
    "aqua_rat": 2000,
    "allenai/openbookqa": 4750,
    "facebook/anli": 2000,
    "ChilleD/StrategyQA": 1580
}
COT_EXAMPLES = {}
COT_EXAMPLES['gsm8k'] = gsm8k_cot_prompts
COT_EXAMPLES['aqua_rat'] = AQuA_cot_prompts
COT_EXAMPLES["ChilleD/SVAMP"] = gsm8k_cot_prompts
COT_EXAMPLES["allenai/openbookqa"] = OBQA_cot_prompts
COT_EXAMPLES["facebook/anli"] = ANLI_cot_prompts
COT_EXAMPLES["ChilleD/StrategyQA"] = StrategyQA_cot_prompts
COT_EXAMPLES["self"] = self_weight_propmts

def derive_ratings_from_answer(answer):
    pattern = r"(\d+\.?\d*)"
    ret = re.search(pattern, answer)
    if ret is None:
        return None
    num = ret.group(1)
    if len(num) > 20:
        num = num[:20]
    return num

def derive_num_from_answer(answer_test):
    num = answer_test.split("####")[-1]
    num = num.strip()
    num = num.replace(",", "")
    # num = float(num)
    # num = int(num)
    return num

def format_ground_truth_answer(ground: str):
    ground = ground.replace("####", "The answer is")
    ground = re.sub(r"<<.*?>>", "", ground)
    return ground

def derive_number_opencompass(text):
    text = text.split("Question:")[0]
    numbers = re.findall(r'\-?\d+\.\d+\-?\d+', text)
    if not numbers:
        return None
    if len(numbers) > 30:
        return None
    return numbers[-1]

def derive_num_from_output(output_text):
    new_text = output_text.lower()
    new_text = new_text.split("question:")[0]
    suffix = new_text.split("the answer is")[-1]
    suffix = suffix.strip()
    if "=" in suffix:
        suffix = suffix.split("=")[-1].strip()
    if len(suffix) <= 0:
        return None
    pattern = r"(\D*?)(\d+\.?\d*)"
    ret = re.search(pattern, suffix.replace(",", ""))
    if ret is None:
        # with open("/home/LAB/jiangcy/AdaDF/error_log.txt","a") as f:
        #     f.write("-------------\n")
        #     f.write(question)
        #     f.write("\n")
        #     f.write(output_text)
        #     f.write("\n")
        return None
    num = ret.group(2)
    if len(num) > 30:
        return None
    return num

def derive_choice_from_output(output_text):
    new_text = output_text.lower()
    suffix = new_text.split("the answer is")[-1]
    suffix = suffix.strip()
    if "=" in suffix:
        suffix = suffix.split("=")[-1].strip()
    if len(suffix) <= 0:
        return None
    pattern = r"\(([a-z])\)"
    ret = re.search(pattern, suffix.replace(",", ""))
    if ret is None:
        # with open("/home/LAB/jiangcy/AdaDF/error_log.txt","a") as f:
        #     f.write("-------------\n")
        #     f.write(question)
        #     f.write("\n")
        #     f.write(output_text)
        #     f.write("\n")
        return None
    choice = ret.group(1)
    if len(choice) > 30:
        return None
    return choice.strip().upper()

def get_qa_pair(question, output_text):
    if question in output_text:
        new_text = output_text.split(question)[1]
    else:
        new_text = output_text
    new_text.strip()
    new_text += "\n"
    if "Q:" not in new_text:
        answer_text = new_text
    else:
        answer_text = new_text.split("Q:")[0]
    answer_text = answer_text.replace("A:", "")
    answer_lower = answer_text.lower()
    if "the answer is" in answer_lower:
        suffix = answer_lower.split("the answer is")[-1]
        pre_cnt = len(answer_text) - len(suffix)
        answer = derive_num_from_output(answer_lower)
        if answer is not None:
            answer_text = answer_text[:pre_cnt] + " {}.".format(str(int(float(answer))))
    answer = answer_text.strip()
    return question, answer

def get_extractors(name):
    if name == "gsm8k":
        dp = lambda x: COT_EXAMPLES["gsm8k"] + "\n" + "Q: " + x["question"] + "\n" + "A: "
        ge = lambda x: derive_num_from_answer(x["answer"])
        qe = lambda x: x["question"]

    elif name == "ChilleD/SVAMP":
        dp = lambda x: COT_EXAMPLES["ChilleD/SVAMP"] + "\n" + "Q: " + x["Body"] + " " + x["Question"] + "\n" + "A: "
        ge = lambda x: int(float(str(x["Answer"])))
        qe = lambda x: x["Body"] + " " + x["Question"]

    elif name == "aqua_rat":
        dp = lambda x: COT_EXAMPLES["aqua_rat"] + "\n" + "Q: " + x["question"] + " Answer Choices are:" + "; ".join(
            x["options"]) + "\n" + "A: "
        ge = lambda x: x["correct"]
        qe = lambda x: x["question"] + " Answer Choices are:" + "; ".join(x["options"])

    elif name == "allenai/openbookqa":
        dp = lambda x: COT_EXAMPLES["allenai/openbookqa"] + "\n" + \
                       "Q: " + x["question_stem"] + (
                           "" if x["question_stem"].strip().endswith("?") else "?") + " Answer Choices are: " + \
                       "; ".join(["{}){}".format(l, t) for l, t in
                                  zip(x["choices"]["label"], x["choices"]["text"])]) + ".\n" + "A: "
        ge = lambda x: x["answerKey"]
        qe = lambda x: x["question_stem"] + (
            "" if x["question_stem"].strip().endswith("?") else "?") + " Answer Choices are: " + \
                       "; ".join(["{}){}".format(l, t) for l, t in zip(x["choices"]["label"], x["choices"]["text"])])

    elif name == "facebook/anli" or name == "facebook/anli2":
        dp = lambda x: COT_EXAMPLES["facebook/anli"] + "\n" + \
                       'Q: ' + '"' + x["premise"] + '" Based on this premise, can we conclude the hypothesis "' + \
                       x[
                           "hypothesis"] + '" is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.' + "\n" + "A: "
        ge = lambda x: "A" if int(x["label"]) == 0 else "B" if int(x["label"]) == 2 else "C"
        qe = lambda x: '"' + x["premise"] + '" Based on this premise, can we conclude the hypothesis "' + \
                       x["hypothesis"] + '" is true?\nAnswer Choices are: A)yes; B)no; C)impossible to tell.'
    elif name == "ChilleD/StrategyQA":
        dp = lambda x: COT_EXAMPLES["ChilleD/StrategyQA"] + "\n" + \
                       "Q: " + x["question"] + " Answer Choices are: A)yes; B)no." + "\n" + "A:"
        ge = lambda x: "A" if x["answer"] else "B"
        qe = lambda x: x["question"] + " Answer Choices are: A)yes; B)no."
    else:
        print("wrong name!")
        quit()
    return dp, ge, qe

def qa_preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    processor,
    system_message: str = ""
):
    input_ids = []
    targets = []
    masks = []
    # input_text = ["Q: " + source["question"] + tokenizer.eos_token + "A: " + source["answer"] for source in sources]
    input_text = [processor(source) for source in sources]
    encoding = tokenizer(
        input_text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors='pt'
    )
    input_ids = encoding["input_ids"]
    targets = copy.deepcopy(encoding["input_ids"])
    masks = encoding['attention_mask']
    return dict(
        input_ids=input_ids,
        labels=targets,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
        attention_mask=masks,
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, data_processor, weight_extractor=None, groundtruth_extractor=None, question_extractor=None):
        super(SupervisedDataset, self).__init__()

        print("Formatting inputs...")
        # sources = [example["conversations"] for example in raw_data]
        self.sources = raw_data
        if weight_extractor is not None:
            self.data_weights = [weight_extractor(d) for d in self.sources]
        else:
            self.data_weights = [0] * len(self.sources)
        if groundtruth_extractor is not None:
            self.groundtruths = [groundtruth_extractor(d) for d in self.sources]
        else:
            self.groundtruths = [0] * len(self.sources)
        if question_extractor is not None:
            self.questions = [question_extractor(d) for d in self.sources]
        else:
            self.questions = [0] * len(self.sources)
        data_dict = qa_preprocess(self.sources, tokenizer, max_len, data_processor)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids = self.input_ids[i],
            labels = self.labels[i],
            attention_mask = self.attention_mask[i],
            weight = self.data_weights[i],
            groundtruth = self.groundtruths[i],
            question = self.questions[i]
        )
    

def build_dataset(name:str):
    if name == "gsm8k":
        valid_dataset = load_dataset("gsm8k", "main", split="train[:25]")
        valid_dataset = [
            [
                dict(role='user', content="Question: {}\nLet's think step by step\nAnswer:".format(data["question"])),
                dict(role='assistant', content='{}'.format(format_ground_truth_answer(data["answer"]))),
            ]
            for data in valid_dataset
        ]
        return valid_dataset
    if name == "math":
        datasets = [
            load_dataset("lighteval/MATH", "algebra", split="train[:5]"),
            load_dataset("lighteval/MATH", "counting_and_probability", split="train[:5]"),
            load_dataset("lighteval/MATH", "geometry", split="train[:5]"),
            load_dataset("lighteval/MATH", "intermediate_algebra", split="train[:5]"),
            load_dataset("lighteval/MATH", "number_theory", split="train[:5]"),
            load_dataset("lighteval/MATH", "prealgebra", split="train[:5]"),
            load_dataset("lighteval/MATH", "precalculus", split="train[:5]")
        ]
        valid_dataset = []
        for ds in datasets:
            valid_dataset += [
                [
                    dict(role='user', content="Question: {}\nLet's think step by step\nAnswer:".format(data["problem"])),
                    dict(role='assistant', content='{}'.format(data["solution"])),
                ]
                for data in ds
            ]
        return valid_dataset
    if name == "openai_humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test[:10]")
        valid_dataset = [
            [
                dict(role='user', content="Question: {}\nLet's think step by step\nAnswer:".format(data["prompt"])),
                dict(role='assistant', content='{}'.format(data["canonical_solution"])),
            ]
            for data in dataset
        ]
        return valid_dataset
    if name == "IFEval":
        pass

GSM8K_SET = build_dataset("gsm8k")
MATH_SET = build_dataset("math")
OPENAI_SET = build_dataset("openai_humaneval")


def compute_loss(model, tokenizer, prompts, device):
    model.eval()
    tokenizer.padding_side = "left"
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    inputs = [
        "\n".join([m[0]["content"], m[1]["content"]])
        for m in prompts
    ]
    total_loss = []
    i = 0
    while i < len(inputs):
        batch_inputs = tokenizer(
            inputs[i:i+8],
            # inputs,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=2048,
            add_special_tokens=True).to(device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model(**batch_inputs)
            logits = outputs.get("logits")
            labels = copy.deepcopy(batch_inputs["input_ids"])
            batch_size = logits.shape[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view(batch_size, -1)
        mask = batch_inputs["attention_mask"][:, 1:]
        loss = loss * mask
        loss = torch.sum(loss, dim=-1)
        divider = torch.sum(mask, dim=-1)
        loss = torch.div(loss, divider).detach().cpu().numpy()
        avg_loss = np.mean(loss)
        total_loss.append(avg_loss)
        i += 8
    return sum(total_loss)/len(total_loss)
    # return avg_loss

def compute_metrics(model, tokenizer, device="cpu"):
    gsm8k_loss = compute_loss(model, tokenizer, GSM8K_SET, device)
    math_loss = compute_loss(model, tokenizer, MATH_SET, device)
    humaneval = compute_loss(model, tokenizer, OPENAI_SET, device)
    return (gsm8k_loss + math_loss + humaneval)/3