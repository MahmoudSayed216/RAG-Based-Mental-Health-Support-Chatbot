prompt = "from {src_lang} to {dst_lang}"

arguments = {
    '{src_lang}': 'arabic',
    '{dst_lang}': 'english'
}

for key, val in arguments.items():
    prompt = prompt.replace(key, val)


print(prompt)