import pathlib
import json


def count_all():
    # Load the JSON file
    folder_path = pathlib.Path(__file__).parent
    count = 0
    for file_path in folder_path.glob('*.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
            count += len(data)
    print(f'Total number of queries across all JSON files: {count}')


def get_questions():
    # Load the JSON file
    folder_path = pathlib.Path(__file__).parent
    questions = []
    statements = []
    for file_path in folder_path.glob('*.json'):
        if 'templates.json' in file_path.name:
            with open(file_path, 'r') as file:
                data = json.load(file)
                for item in data:
                    if item.endswith('?'):
                        questions.append(item)
                    else:
                        statements.append(item)
    # Save the questions to a new JSON file
    json_path = folder_path / 'templates-general-questions.json'
    with open(json_path, 'w') as file:
        json.dump(questions, file, indent=2, ensure_ascii=False)
    # Save the statements to a new JSON file
    json_path = folder_path / 'templates-general-statements.json'
    with open(json_path, 'w') as file:
        json.dump(statements, file, indent=2, ensure_ascii=False)
    return questions, statements

if __name__ == "__main__":
    count_all()
    # questions, statements = get_questions()
    # print(f"Total number of queries: {len(questions) + len(statements)}")
    # print(f"Questions: {len(questions)}")
    # print(f"Statements: {len(statements)}")
    # print(f"Total number of queries in templates-scraped-questions.json: {len(questions) + len(statements)}")