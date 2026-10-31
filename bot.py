import os
import json
import logging
import asyncio
import re
import secrets
import time
from html import unescape
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
log = logging.getLogger("instruktor-bot")
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
QUESTION_INTERVAL_MINUTES = int(os.getenv("QUESTION_INTERVAL_MINUTES", "30"))

LEGAL_SOURCES = [
    {
        "title": "Ustawa o kierujących pojazdami — tekst jednolity, Dz.U. 2024 poz. 1210",
        "url": "https://eli.gov.pl/api/acts/DU/2024/1210/text.html",
    },
    {
        "title": "Prawo o ruchu drogowym — oficjalny tekst ELI (adres konfigurowalny)",
        "url": os.getenv("PRAWO_O_RUCHU_DROGOWYM_URL", "https://eli.gov.pl/api/acts/DU/2024/1251/text.html"),
    },
    {
        "title": "Rozporządzenie w sprawie szkolenia osób ubiegających się o uprawnienia do kierowania pojazdami, instruktorów i wykładowców",
        "url": os.getenv("SZKOLENIE_ROZPORZADZENIE_URL", "https://eli.gov.pl/api/acts/DU/2016/280/text.html"),
    },
    {
        "title": "Rozporządzenie w sprawie uzyskiwania uprawnień przez instruktorów i wykładowców — Dz.U. 2019 poz. 1200",
        "url": "https://eli.gov.pl/api/acts/DU/2019/1200/text.html",
    },
]



class HealthHandler(BaseHTTPRequestHandler):
    def _send_health_response(self, include_body=True):
        if self.path not in ("/", "/health"):
            self.send_response(404)
            self.end_headers()
            return

        body = b"OK"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        if include_body:
            self.wfile.write(body)

    def do_GET(self):
        self._send_health_response(include_body=True)

    def do_HEAD(self):
        self._send_health_response(include_body=False)

    def log_message(self, format, *args):
        pass


def start_health_server():
    port = int(os.environ.get("PORT", "10000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    log.info("Health server listening on port %s", port)


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = unescape(soup.get_text(" ", strip=True))
    return re.sub(r"\s+", " ", text)


def fetch_legal_context() -> str:
    blocks = []
    for source in LEGAL_SOURCES:
        try:
            parsed = urlparse(source["url"])
            if parsed.scheme != "https" or parsed.netloc != "eli.gov.pl":
                log.warning("Skipping non-ELI legal source: %s", source["url"])
                continue
            response = requests.get(
                source["url"], timeout=15,
                headers={"User-Agent": "InstruktorExamTrainer/1.0"},
            )
            response.raise_for_status()
            text = clean_text(response.text)
            # Bound the prompt size; this does not guarantee the relevant provision is included.
            blocks.append(
                f"ŹRÓDŁO: {source['title']}\nURL: {source['url']}\nTEKST: {text[:26000]}"
            )
        except Exception as exc:
            log.warning("Could not fetch legal source %s: %s", source["url"], exc)
    if not blocks:
        raise RuntimeError(
            "Nie udało się pobrać żadnego tekstu z oficjalnego ELI. "
            "Nie generuję pytania bez podstawy prawnej."
        )
    return "\n\n---\n\n".join(blocks)


def generate_question() -> dict:
    legal_context = fetch_legal_context()
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
Jesteś autorem trudnych, rzetelnych pytań treningowych dotyczących wyłącznie
przepisów ruchu drogowego w Polsce.

Przygotuj jedno wymagające pytanie oparte na dostarczonych tekstach oficjalnych aktów
prawnych. Nie wybieraj pytania według z góry ustalonej listy tematów ani rotacji kategorii.
Skup się na trudnych przypadkach, wyjątkach od reguł, subtelnych rozróżnieniach prawnych,
wiarygodnie brzmiących błędnych odpowiedziach oraz częstych nieporozumieniach pojęciowych.
Pytanie powinno opisywać konkretną sytuację/scenariusz, jeśli materiał prawny na to pozwala.
Nie twierdź, że dany błąd rzeczywiście jest częsty na egzaminie, jeśli dostarczone źródła
nie zawierają takich danych. Nie przedstawiaj pytania jako oficjalnego pytania egzaminacyjnego.

Zakres tematyczny jest ściśle ograniczony do przepisów ruchu drogowego i praktycznych
sytuacji na drodze. Pytania mogą dotyczyć m.in. pierwszeństwa przejazdu, skrzyżowań,
znaków i sygnałów drogowych, wyprzedzania, omijania, wymijania, prędkości, zatrzymania
i postoju, przejazdów kolejowych, zachowania wobec pieszych i rowerzystów, wykonywania
manewrów oraz zachowania w szczególnych warunkach drogowych.

Bezwzględnie wyklucz pytania dotyczące pedagogiki, psychologii, metodyki nauczania,
prowadzenia zajęć, szkolenia kandydatów na kierowców, kwalifikacji instruktorów,
uzyskiwania prawa jazdy, egzaminowania, ośrodków szkolenia kierowców, starostów,
procedur administracyjnych oraz uprawnień zawodowych instruktorów. Nie twórz pytań
o formalne wymagania dotyczące zawodu instruktora ani o organizację szkolenia.

Wybieraj wyłącznie zagadnienia bezpośrednio związane z ruchem drogowym. Korzystaj
z dostarczonych aktów prawnych jako źródeł przepisów, ale ignoruj ich fragmenty
dotyczące wyżej wykluczonych zagadnień. Jeśli nie możesz oprzeć pytania na konkretnym
przepisie dotyczącym ruchu drogowego, zwróć JSON z polem "error" zamiast tworzyć
pytanie spoza dozwolonego zakresu.

BEZWZGLĘDNE OGRANICZENIE ZAKRESU TEMATYCZNEGO:

Pytania mają sprawdzać wyłącznie znajomość zasad ruchu drogowego oraz prawidłowego
zachowania uczestników ruchu podczas poruszania się po drogach.

NIE WOLNO tworzyć pytań dotyczących:
- punktów karnych, ich naliczania, redukowania, usuwania ani kursów redukujących punkty;
- mandatów, wysokości grzywien i procedur związanych z karaniem kierowców;
- zatrzymywania, cofania ani przywracania prawa jazdy;
- okresu próbnego, limitów punktów i konsekwencji administracyjnych naruszeń;
- uzyskiwania uprawnień, egzaminów, szkoleń i kwalifikacji kierowców;
- starostów, WORD, ewidencji kierowców i procedur administracyjnych;
- pedagogiki, psychologii, metodyki nauczania ani pracy instruktora;
- dokumentów potwierdzających uprawnienia do kierowania pojazdami, w tym
  tymczasowego elektronicznego prawa jazdy, elektronicznego prawa jazdy,
  fizycznego dokumentu prawa jazdy i międzynarodowego prawa jazdy;
- zasad wydawania, odbioru, ważności, przedłużania i zakresu dokumentów
  prawa jazdy;
- momentu uzyskania uprawnień do kierowania pojazdami oraz warunków,
  od których zależy możliwość legalnego rozpoczęcia kierowania pojazdem.
  
Rozróżniaj przepisy regulujące SPOSÓB poruszania się po drodze od przepisów
regulujących PRAWO DO kierowania pojazdem. Dozwolona jest wyłącznie pierwsza
kategoria. Pytania o to, kto, kiedy i na podstawie jakiego dokumentu może
legalnie kierować pojazdem, są niedozwolone.

WAŻNE: Sam fakt, że dane zagadnienie jest uregulowane w ustawie Prawo o ruchu
drogowym lub wynika z naruszenia przepisów ruchu drogowego, NIE oznacza, że
należy ono do dozwolonego zakresu tematycznego.

Przykładowo pytanie o to, czy kierowca posiadający 22 punkty karne może
uczestniczyć w kursie redukującym punkty, jest ZABRONIONE.

PRZED ZWRÓCENIEM JSON sprawdź, czy pytanie dotyczy bezpośrednio zasad poruszania
się po drodze, pierwszeństwa, wykonywania manewrów, znaków, sygnałów, zachowania
uczestników ruchu lub bezpieczeństwa ruchu. Jeśli dotyczy któregokolwiek
z wykluczonych zagadnień albo konsekwencji administracyjnych naruszeń,
odrzuć je i wygeneruj inne pytanie.

Zasady:
- Całość pytania, odpowiedzi i wyjaśnienia napisz po polsku.
- Zwróć dokładnie 4 odpowiedzi oznaczone A, B, C, D.
- Od 1 do 3 odpowiedzi może być prawidłowych; nie ujawniaj liczby prawidłowych odpowiedzi
  w treści pytania.
- Sprawdź precyzyjnie warunki, wyjątki i ograniczenia wynikające z przekazanych przepisów.
- Każde twierdzenie prawne oraz każda wskazana podstawa prawna muszą być poparte tekstem
  dostarczonych źródeł. Nie wymyślaj przepisów, numerów artykułów ani ich brzmienia.
- Jeśli nie możesz przygotować pytania z wiarygodną i konkretną podstawą prawną na podstawie
  dostarczonych tekstów, nie zgaduj: zwróć JSON z polem "error" wyjaśniającym problem.
- Podstawa prawna ma wskazywać pełną nazwę aktu, możliwie dokładny artykuł/ustęp/punkt,
  oficjalny URL ELI oraz krótkie wyjaśnienie, jak przepis uzasadnia odpowiedź.
- Wyjaśnij każdą odpowiedź osobno: dlaczego jest prawidłowa albo błędna. Zwróć uwagę na
  konkretny szczegół scenariusza, który zmienia wynik lub stanowi pułapkę.
- Nie przedstawiaj niezweryfikowanej opinii jako przepisu prawa. Jeśli źródło nie wystarcza,
  wybierz inne pytanie.
- Zwróć wyłącznie poprawny JSON, bez Markdown i bez komentarzy.

Schemat JSON dla poprawnego pytania:
{{
  "question": "treść pytania / scenariusza",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "correct_indices": [0, 2],
  "explanation": "szczegółowe wyjaśnienie po polsku, w tym ocena każdej odpowiedzi",
  "legal_basis": [{{"act": "pełna nazwa aktu", "provision": "art. X ust. Y pkt Z", "url": "https://eli.gov.pl/...", "relevance": "jak przepis uzasadnia odpowiedź"}}],
  "verification_note": "uwaga o ograniczeniu lub konieczności sprawdzenia aktualności; w przeciwnym razie pusty tekst"
}}

Jeśli nie możesz rzetelnie opracować pytania, zwróć wyłącznie:
{{"error": "krótkie wyjaśnienie"}}

TEKSTY Z OFICJALNEGO ELI:
{legal_context}
"""
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.35,
                ),
            )
            break

        except Exception as exc:
            status = getattr(exc, "code", None)
            if status is None:
                status = getattr(exc, "status_code", None)

            if status not in (429, 500, 503) or attempt == 2:
                raise

            delay = 2 ** (attempt + 1)
            log.warning(
                "Gemini returned HTTP %s; retrying in %s seconds "
                "(attempt %s/3)",
                status, delay, attempt + 2,
            )
            time.sleep(delay)

    data = json.loads(response.text)
    if data.get("error"):
        raise ValueError(f"Gemini nie wygenerował pytania: {data['error']}")
    validate_question(data)
    return data


def validate_question(q: dict):
    if not isinstance(q.get("question"), str) or len(q["question"]) < 20:
        raise ValueError("Generated question is missing or too short")
    options = q.get("options")
    if not isinstance(options, list) or len(options) != 4 or not all(isinstance(x, str) for x in options):
        raise ValueError("Question must contain exactly four text options")
    correct = q.get("correct_indices")
    if not isinstance(correct, list) or not 1 <= len(correct) <= 3 or any(type(i) is not int or i not in range(4) for i in correct):
        raise ValueError("Correct answers must be 1-3 valid option indices")
    basis = q.get("legal_basis")
    if not isinstance(basis, list) or not basis:
        raise ValueError("Question must include at least one legal basis reference")
    for ref in basis:
        url = ref.get("url", "")
        if not url.startswith("https://eli.gov.pl/"):
            raise ValueError("Legal basis URL must point to official ELI")


def get_state(context: ContextTypes.DEFAULT_TYPE) -> dict:
    """Transient, in-memory state only; it is lost when the process restarts."""
    state = context.application.bot_data
    state.setdefault("questions", {})
    state.setdefault("answers", {})
    state.setdefault("checked", set())
    return state


def format_question(q: dict) -> str:
    options = "\n".join(q["options"])
    return (
        "🧠 <b>Pytanie treningowe — instruktor nauki jazdy</b>\n\n"
        f"{escape_html(q['question'])}\n\n{escape_html(options)}\n\n"
        "Wybierz wszystkie odpowiedzi, które uważasz za prawidłowe, "
        "a następnie naciśnij <b>Sprawdź odpowiedź</b>."
    )


def escape_html(text: str) -> str:
    import html
    return html.escape(str(text))


def question_keyboard(question_id: str, selected: set[int]) -> InlineKeyboardMarkup:
    letters = ["A", "B", "C", "D"]
    row = [
        InlineKeyboardButton(
            ("☑ " if i in selected else "☐ ") + letters[i],
            callback_data=f"toggle:{question_id}:{i}",
        )
        for i in range(4)
    ]
    return InlineKeyboardMarkup([
        row,
        [InlineKeyboardButton("Sprawdź odpowiedź", callback_data=f"check:{question_id}")],
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    context.application.bot_data["chat_id"] = chat_id
    context.application.bot_data["scheduled_enabled"] = True
    await update.message.reply_text(
        "Cześć! Będę wysyłać pytania treningowe na instruktora nauki jazdy co "
        f"{QUESTION_INTERVAL_MINUTES} minut.\n\n"
        "/now — nowe pytanie teraz\n"
        "/stop — wyłącz regularne pytania\n"
        "/start — włącz ponownie regularne pytania\n\n"
        "Po każdym restarcie bota wyślij /start, aby ponownie włączyć zaplanowane pytania.\n\n"
        "Pytania są generowane przez AI na podstawie pobranych tekstów z ELI. "
        "Sprawdzaj wskazane przepisy w aktualnym brzmieniu; pytania nie są oficjalnymi "
        "pytaniami egzaminacyjnymi."
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data["scheduled_enabled"] = False
    await update.message.reply_text(
        "Regularne pytania zostały wyłączone. Użyj /start, aby je ponownie włączyć. "
        "Polecenie /now nadal działa."
    )


async def send_question_to_chat(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    try:
        q = await asyncio.to_thread(generate_question)
        question_id = secrets.token_hex(4)
        state = get_state(context)
        state["questions"][question_id] = q
        while len(state["questions"]) > 100:
            oldest_id = next(iter(state["questions"]))
            state["questions"].pop(oldest_id, None)
            state["checked"].discard(oldest_id)
            for key in list(state["answers"]):
                if key[0] == oldest_id:
                    state["answers"].pop(key, None)
        await context.bot.send_message(
            chat_id=chat_id,
            text=format_question(q),
            parse_mode=ParseMode.HTML,
            reply_markup=question_keyboard(question_id, set()),
            disable_web_page_preview=True,
        )
    except Exception:
        log.exception("Could not generate/send question to %s", chat_id)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Nie udało się teraz przygotować pytania (problem z AI lub źródłem prawnym). "
                 "Spróbuję przy kolejnej okazji. Możesz też użyć /now później.",
        )


async def now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Przygotowuję pytanie…")
    await send_question_to_chat(update.effective_chat.id, context)


async def scheduled_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.application.bot_data.get("chat_id")
    if not context.application.bot_data.get("scheduled_enabled", False):
        log.info("Scheduled questions are disabled; skipping this interval.")
        return
    if chat_id is None:
        log.info("No chat registered yet. Send /start to enable scheduled questions.")
        return
    try:
        await send_question_to_chat(int(chat_id), context)
    except Exception:
        log.exception("Could not send scheduled question")


async def send_long_message(message, text, limit=4000, parse_mode=None):
    while text:
        if len(text) <= limit:
            chunk = text
            text = ""
        else:
            split_at = text.rfind("\n", 0, limit)
            if split_at == -1:
                split_at = limit
            chunk = text[:split_at]
            text = text[split_at:].lstrip()

        await message.reply_text(chunk, parse_mode=parse_mode)


async def answer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    parts = query.data.split(":")
    action = parts[0]
    question_id = parts[1]
    chat_id = query.message.chat_id
    state = get_state(context)
    q = state["questions"].get(question_id)
    if not q:
        await query.message.reply_text(
            "To pytanie nie jest już dostępne (np. po restarcie bota). Użyj /now, aby wygenerować nowe."
        )
        return

    answer_key = (question_id, chat_id)
    current = state["answers"].setdefault(answer_key, set())

    if action == "toggle":
        if question_id in state["checked"]:
            await query.answer("Ta odpowiedź została już sprawdzona.", show_alert=True)
            return
        idx = int(parts[2])
        if idx in current:
            current.remove(idx)
        else:
            current.add(idx)
        await query.edit_message_reply_markup(
            reply_markup=question_keyboard(question_id, current)
        )
        return

    if action == "check":
        if question_id in state["checked"]:
            await query.message.reply_text("Ta odpowiedź została już sprawdzona.")
            return
        correct = set(q["correct_indices"])
        is_correct = current == correct
        state["checked"].add(question_id)
        answer_letters = ", ".join("ABCD"[i] for i in sorted(correct))
        basis_lines = []
        for ref in q.get("legal_basis", []):
            provision = ref.get("provision", "przepis do sprawdzenia")
            act = ref.get("act", "Akt prawny")
            url = ref.get("url", "")
            relevance = ref.get("relevance", "")
            line = f"• {escape_html(act)} — {escape_html(provision)}"
            if relevance:
                line += f"\n{escape_html(relevance)}"
            if url:
                line += f"\n{escape_html(url)}"
            basis_lines.append(line)
        result = "✅ <b>Poprawnie!</b>" if is_correct else "❌ <b>Nie całkiem.</b>"
        explanation = (
            f"{result}\nPrawidłowe odpowiedzi: <b>{answer_letters}</b>\n\n"
            f"<b>Wyjaśnienie</b>\n{escape_html(q.get('explanation', 'Brak wyjaśnienia.'))}\n\n"
            f"<b>Podstawa prawna (sprawdź aktualne brzmienie)</b>\n" + "\n".join(basis_lines)
        )
        note = q.get("verification_note", "")
        if note:
            explanation += f"\n\n<b>Uwaga:</b> {escape_html(note)}"
        explanation += "\n\n<i>To pytanie treningowe wygenerowane przez AI, nie oficjalne pytanie egzaminacyjne.</i>"
        await send_long_message(
            query.message,
            explanation,
            limit=4000,
            parse_mode=ParseMode.HTML
        )
        await query.edit_message_reply_markup(reply_markup=None)


def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("now", now))
    app.add_handler(CallbackQueryHandler(answer_callback, pattern=r"^(toggle|check):"))
    if app.job_queue is None:
        raise RuntimeError("JobQueue is unavailable. Install python-telegram-bot[job-queue].")
    app.job_queue.run_repeating(
        scheduled_question,
        interval=QUESTION_INTERVAL_MINUTES * 60,
        first=15,
        name="scheduled-questions",
    )
    log.info("Starting instructor exam trainer; interval=%s minutes", QUESTION_INTERVAL_MINUTES)
    log.info("Starting instructor exam trainer; interval=%s minutes", QUESTION_INTERVAL_MINUTES)

    start_health_server()

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
