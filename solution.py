
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from rag import EmbeddingIndex, Fragment, ask_deepseek, build_embedding_index, load_and_split_documents, search


ANSWER_LIMIT = 10_000
REQUEST_WINDOW_SECONDS = 18 * 60
FALLBACK = "Не удалось получить ответ из-за технической ошибки."


def read_int(name: str, default: int, upper: int) -> int:
    try:
        return max(1, min(upper, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def find_input(data_dir: Path, *names: str) -> Path:
    for name in names:
        path = data_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"В {data_dir} не найден ни один из файлов: {', '.join(names)}")


def read_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("Во входном CSV нужны колонки id и question.")
        questions = [(row["id"], row["question"]) for row in reader]

    seen = set()
    for question_id, question in questions:
        if not question_id or not question_id.strip() or question is None:
            raise ValueError("Во входном CSV есть строка без ID или поля question.")
        if question_id in seen:
            raise ValueError("Во входном CSV повторяется ID.")
        seen.add(question_id)
    return questions


def answer_question(
    row: tuple[str, str],
    *,
    fragments: list[Fragment],
    embedding_index: EmbeddingIndex | None,
    top_k: int,
    deadline: float,
) -> str:
    question_id, question = row
    if time.monotonic() >= deadline:
        return FALLBACK
    found = search(question, fragments, top_k=top_k, embedding_index=embedding_index)
    for attempt in range(2):
        try:
            answer = ask_deepseek(question, found)
            if len(answer) > ANSWER_LIMIT:
                return answer[: ANSWER_LIMIT - 1].rstrip() + "…"
            return answer
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            retryable = status == 429 or 500 <= status < 600
            print(f"Вопрос {question_id}: API вернул HTTP {status}.", file=sys.stderr)
        except httpx.TransportError:
            retryable = True
            print(f"Вопрос {question_id}: ошибка соединения с API.", file=sys.stderr)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            retryable = False
            print(f"Вопрос {question_id}: ошибка конфигурации или ответа API ({type(error).__name__}).", file=sys.stderr)

        if not retryable or attempt == 1 or time.monotonic() + 1 >= deadline:
            break
        time.sleep(1)

    return FALLBACK


def write_submission(questions: list[tuple[str, str]], answers: list[str]) -> None:
    output = Path("submission.csv")
    with NamedTemporaryFile(
        mode="w", dir=output.parent, encoding="utf-8", newline="", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["id", "answer"])
            for (question_id, _), answer in zip(questions, answers, strict=True):
                writer.writerow([question_id, answer])
            stream.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    started = time.monotonic()
    deadline = started + REQUEST_WINDOW_SECONDS
    data_dir = Path(os.getenv("DATA_DIR") or "data")
    questions = read_questions(find_input(data_dir, "questions.csv", "demo_questions.csv"))
    fragments = load_and_split_documents(
        find_input(data_dir, "knowledge-base.zip", "kb.zip", "knowledge_base.zip")
    )
    workers = read_int("MAX_WORKERS", default=2, upper=8)
    # Связанные положения могут занимать несколько мест в выдаче.
    top_k = max(6, read_int("TOP_K", default=6, upper=30))
    print(f"Вопросов: {len(questions)}. Фрагментов: {len(fragments)}.", flush=True)

    index_started = time.monotonic()
    embedding_index = build_embedding_index(fragments)
    print(f"Построение индекса: {time.monotonic() - index_started:.1f} с.", flush=True)
    search_mode = "слова и эмбеддинги" if embedding_index is not None else "слова"
    print(f"Режим поиска: {search_mode}.", flush=True)
    process = partial(
        answer_question,
        fragments=fragments,
        embedding_index=embedding_index,
        top_k=top_k,
        deadline=deadline,
    )
    answers = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for position, answer in enumerate(pool.map(process, questions), start=1):
            answers.append(answer)
            if position % 10 == 0 or position == len(questions):
                print(f"Обработано {position}/{len(questions)}.", flush=True)

    write_submission(questions, answers)
    print(f"Создан submission.csv: {len(answers)} ответов.", flush=True)
    print(f"Технических заглушек: {answers.count(FALLBACK)} из {len(answers)}.", flush=True)
    print(f"Время работы: {time.monotonic() - started:.1f} с.", flush=True)


if __name__ == "__main__":
    main()
