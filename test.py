import csv
import random
from itertools import chain

from docutils.languages.af import labels

from generate_data import add_noise

# 数据生成模板
templates = {
    "SQLi": [
        "'{user}' {logic} 1=1-- -",
        "' {union} SELECT null,{func}--",
        "'; {danger_cmd}--",
        "' {logic} EXISTS(SELECT * FROM {table})--"
    ],
    "Command-Injection": [
        "& {unix_cmd}",
        "; {unix_cmd}",
        "| {unix_cmd}",
        "`{unix_cmd}`",
        "$({danger_cmd})",
        "<% {java_cmd} %>"
    ],
    "Path-Traversal": [
        "/{dir}/%2e%2e/%2e%2e/{file}",
        "../../../../{file}",
        "....//....//{win_file}",
        "%252e%252e%252f{enc_file}",
        "C:%5C..%5C..%5C{win_file}"
    ],
    "XSS": [
        "<{tag} {event}=alert(1)>",
        "<{tag} src=x onerror={func}>",
        "<iframe src=javascript:{code}>",
        "<object data=data:text/html;base64,{b64}>"
    ],
    "Normal": [
        "{category}/search?query={word}",
        "{page_type}?page={num}&sort={field}",
        "user/{action}?id={num}",
        "{wp_dir}/uploads/{file}"
    ]
}

# 填充参数
params = {
    "user": ["juanGarcia", "karenLee", "morgan57", "rayMond", "nancy60"],
    "logic": ["OR", "oR", "Or", "||", "aNd", "AnD"],
    "union": ["UNION", "uNion", "UnIoN"],
    "func": ["version()", "database()", "@@version", "user()", "pg_sleep(5)"],
    "danger_cmd": ["DROP TABLE users", "SHUTDOWN", "DELETE FROM logs"],
    "unix_cmd": ["echo $PATH", "cat /etc/shadow", "rm -rf /",
                 "ping -c 3 example.com", "curl http://malicious.site"],
    "dir": ["var/www", "opt/app", "usr/local"],
    "file": ["etc/passwd", "etc/shadow", "proc/self/environ"],
    "win_file": ["boot.ini", "autoexec.bat", "windows/win.ini"],
    "enc_file": ["etc/passwd", "etc/group", "proc/cmdline"],
    "tag": ["img", "svg", "body", "iframe", "object"],
    "event": ["onload", "onerror", "onpageshow", "onmouseover"],
    "code": ["eval(atob('YWxlcnQoJ1hTUycp'))", "document.write('<script>alert(1)</script>')"],
    "b64": ["PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pj4=", "PGJvZHkgb25sb2FkPWFsZXJ0KDEpPg=="],
    "category": ["category", "posts", "blog", "tag", "list"],
    "page_type": ["posts", "articles", "products", "items"],
    "field": ["date", "title", "price", "rating"],
    "action": ["profile", "settings", "history", "preferences"],
    "wp_dir": ["wp-content", "wp-includes", "wp-admin"],
    "word": ["test", "hello+world", "risk", "material"],
    "num": lambda: random.randint(1, 100),
    "table": ["users", "accounts", "logs", "transactions"]
}


def generate_samples(label, count):
    samples = []
    for _ in range(count):
        template = random.choice(templates[label])

        # 动态填充参数
        filled = []
        for segment in template.split():
            if segment.startswith("{"):
                key = segment[1:-1]
                if key == "num":
                    filled.append(str(params["num"]()))
                else:
                    filled.append(random.choice(params.get(key, [""])))
            else:
                filled.append(segment)

        payload = " ".join(filled)
        samples.append((payload, label))
    return samples


# 生成数据集（可根据需要调整数量）
dataset = list(chain(
    generate_samples("SQLi", 35),
    generate_samples("Command-Injection", 20),
    generate_samples("Path-Traversal", 15),
    generate_samples("XSS", 20),
    generate_samples("Normal", 10)
))

# 打乱顺序并添加特殊样本
random.shuffle(dataset)
dataset.extend([
    ("<script>alert('XSS')</script>", "XSS"),
    ("../../../etc/passwd", "Path-Traversal"),
    ("' OR 'a'='a'--", "SQLi"),
    ("; ls -la", "Command-Injection"),
    ("wp-content/uploads/image.jpg", "Normal")
])

# 写入CSV文件
with open('web_vul_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["payload", "label"])
    writer.writerows(dataset)

print("数据集生成完成！共生成样本数:", len(dataset))
