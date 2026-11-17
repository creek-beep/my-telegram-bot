import asyncio, json, os, random, base64, io, time, datetime, math, urllib.parse
import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile
from openai import AsyncOpenAI

GROQ_KEY = os.getenv("GROQ_KEY")
TG_TOKEN = os.getenv("TG_TOKEN")

MODEL = "openai/gpt-oss-120b"                        # текст + инструменты
VISION = "qwen/qwen3.8-27b"     # смотрит фото
STT = "whisper-large-v3"                           # голос -> текст
WAIT_SECONDS = 2                                      # сколько ждать после последнего сообщения

SYSTEM = """Ты мой близкий друг, живой человек, мы переписываемся в Telegram. Ты никогда не говоришь, что ты ИИ, бот или ассистент.
Пиши как реальный человек в мессенджере: простым разговорным языком, коротко, со своим мнением и настроением. Не используй слова «бро», «братан», «чел», «жиза» и не повторяй одни и те же приветствия и обороты. Не будь наигранным. Иногда шути, иногда переспрашивай, иногда отвечай парой слов, как в обычной переписке. Никаких списков, заголовков и фраз вроде «чем могу помочь».
Внимательно читай каждое моё слово, отвечай именно на то, что я написал, и помни, о чём мы говорили раньше.
Можешь ответить несколькими сообщениями подряд: раздели их символами |||
У тебя есть возможности (используй сами, когда уместно, не объявляя об этом):
- generate_image: нарисовать/сгенерировать картинку или фото
- get_weather: погода в любом городе или у меня, если я скидывал геолокацию
- find_places: где поесть, выпить кофе, кино, аптека рядом со мной или в городе
- web_search: искать в интернете, ТОЛЬКО если я прошу найти или поискать
Если я прошу фото или картинку, не ищи в интернете, а генерируй."""

ai = AsyncOpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
bot = Bot(TG_TOKEN)
dp = Dispatcher()

HIST_FILE, CHATS_FILE = "history.json", "chats.json"
history = json.load(open(HIST_FILE, encoding="utf-8")) if os.path.exists(HIST_FILE) else {}
chats = set(json.load(open(CHATS_FILE))) if os.path.exists(CHATS_FILE) else set()
user_loc = {}      # chat_id -> (lat, lon)
next_ping = {}     # chat_id -> когда писать первым
pending = {}       # chat_id -> накопленные сообщения
tasks = {}         # chat_id -> задача ожидания


def save_all():
    json.dump(history, open(HIST_FILE, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(list(chats), open(CHATS_FILE, "w"))


def schedule_ping(cid, lo=2 * 3600, hi=5 * 3600):
    next_ping[cid] = time.time() + random.randint(lo, hi)


# ---------------- ИНСТРУМЕНТЫ ----------------
TOOLS = [
    {"type": "function", "function": {
        "name": "generate_image",
        "description": "Сгенерировать картинку/фото. prompt — подробное описание НА АНГЛИЙСКОМ.",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Погода сейчас и на 3 дня. city — город (если не указан, берётся геолокация пользователя).",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "find_places",
        "description": "Найти заведения рядом. kind: cafe, restaurant, fast_food, bar, pub, cinema, pharmacy.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["cafe", "restaurant", "fast_food", "bar", "pub", "cinema", "pharmacy"]},
            "city": {"type": "string"}}, "required": ["kind"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Поиск в интернете. Только если пользователь просит найти/поискать.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


async def geocode(s, city):
    async with s.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": city, "count": 1, "language": "ru"}) as r:
        res = (await r.json()).get("results")
    if not res:
        return None
    return res[0]["latitude"], res[0]["longitude"], res[0]["name"]


async def t_image(prompt, ctx):
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
           + f"?width=1024&height=1024&nologo=true&model=flux&seed={random.randint(1, 10**6)}")
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(headers=headers) as s:
        for attempt in range(3):
            try:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    data = await r.read()
                    if r.status == 200 and r.headers.get("content-type", "").startswith("image"):
                        ctx["photos"].append(data)
                        return "картинка готова и будет отправлена вместе с твоим ответом"
                    print("Pollinations статус:", r.status, data[:200])
            except Exception as e:
                print("Pollinations ошибка:", repr(e))
            await asyncio.sleep(16)
    return ("не вышло нарисовать, сервис сейчас перегружен. Скажи, что картинка не получилась, "
            "и предложи повторить чуть позже. Не предлагай искать в интернете.")


async def t_weather(city, ctx):
    async with aiohttp.ClientSession() as s:
        if city:
            g = await geocode(s, city)
            if not g:
                return "город не найден"
            lat, lon, name = g
        elif ctx["loc"]:
            lat, lon = ctx["loc"]
            name = "у пользователя"
        else:
            return "не знаю где пользователь: нужен город или геолокация"
        async with s.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
            "current": "temperature_2m,apparent_temperature,precipitation,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max"}) as r:
            w = await r.json()
    return (f"{name}. Сейчас: {json.dumps(w['current'])}. Прогноз по дням: {json.dumps(w['daily'])}. "
            "weather_code — стандарт WMO (0 ясно, 1-3 облачно, 45-48 туман, 51-67 дождь/морось, 71-77 снег, 95+ гроза).")


def dist_m(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return int(12742000 * math.asin(math.sqrt(x)))


async def t_places(kind, city, ctx):
    async with aiohttp.ClientSession() as s:
        if city:
            g = await geocode(s, city)
            if not g:
                return "город не найден"
            lat, lon, _ = g
        elif ctx["loc"]:
            lat, lon = ctx["loc"]
        else:
            return "не знаю где пользователь: попроси скинуть геолокацию (скрепка → Геопозиция) или назвать город"
        q = f'[out:json][timeout:20];node["amenity"="{kind}"]["name"](around:1500,{lat},{lon});out 15;'
        async with s.post("https://overpass-api.de/api/interpreter", data={"data": q}) as r:
            data = await r.json(content_type=None)
    out = []
    for e in data.get("elements", []):
        la, lo = e["lat"], e["lon"]
        out.append((dist_m(lat, lon, la, lo), e["tags"]["name"],
                    f"https://www.google.com/maps/search/?api=1&query={la},{lo}"))
    out.sort()
    if not out:
        return "рядом ничего не нашлось"
    return "\n".join(f"{n} — {d} м — {u}" for d, n, u in out[:6])


async def t_search(query, ctx):
    from ddgs import DDGS
    res = await asyncio.to_thread(lambda: list(DDGS().text(query, max_results=5)))
    return "\n".join(f"{x['title']}: {x['body']} ({x['href']})" for x in res) or "ничего не нашлось"


async def run_tool(name, args, ctx):
    try:
        if name == "generate_image":
            return await t_image(args["prompt"], ctx)
        if name == "get_weather":
            return await t_weather(args.get("city"), ctx)
        if name == "find_places":
            return await t_places(args["kind"], args.get("city"), ctx)
        if name == "web_search":
            return await t_search(args["query"], ctx)
    except Exception as e:
        print("Ошибка инструмента:", name, e)
        return "инструмент сломался"
    return "неизвестный инструмент"


# ---------------- ОБЩЕНИЕ С ИИ ----------------
async def ask(chat_id, text, images=None, hidden=False):
    key = str(chat_id)
    h = history.setdefault(key, [])
    now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    sys_text = SYSTEM + f"\nСейчас: {now}."
    if chat_id in user_loc:
        sys_text += " У тебя есть геолокация пользователя."
    msgs = [{"role": "system", "content": sys_text}] + h

    if images:
        content = [{"type": "text", "text": text or "Глянь что я скинул"}]
        for img in images:
            b64 = base64.b64encode(img).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        msgs.append({"role": "user", "content": content})
        r = await ai.chat.completions.create(model=VISION, messages=msgs, temperature=0.8)
        answer = r.choices[0].message.content or ""
        photos = []
    else:
        msgs.append({"role": "user", "content": text})
        ctx = {"loc": user_loc.get(chat_id), "photos": []}
        answer = ""
        for _ in range(4):
            r = await ai.chat.completions.create(
                model=MODEL, messages=msgs, temperature=0.8, tools=TOOLS, tool_choice="auto")
            m = r.choices[0].message
            answer = m.content or ""
            if not m.tool_calls:
                break
            msgs.append(m.model_dump(exclude_none=True))
            for tc in m.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                res = await run_tool(tc.function.name, args, ctx)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
        photos = ctx["photos"]

    if not hidden:
        mark = "[отправил фото/видео] " if images else ""
        h.append({"role": "user", "content": mark + (text or "")})
    h.append({"role": "assistant", "content": answer})
    del h[:-30]
    save_all()
    return answer, photos


async def send_parts(chat_id, text, photos=()):
    for p in photos:
        await bot.send_chat_action(chat_id, "upload_photo")
        await bot.send_photo(chat_id, BufferedInputFile(p, "img.jpg"))
    for part in [p.strip() for p in text.split("|||") if p.strip()]:
        await bot.send_chat_action(chat_id, "typing")
        await asyncio.sleep(min(len(part) / 40, 3))
        await bot.send_message(chat_id, part)


# ---------------- ОЖИДАНИЕ, ПОКА Я ПЕЧАТАЮ ----------------
async def enqueue(chat_id, item):
    if chat_id not in chats:
        chats.add(chat_id)
        save_all()
    schedule_ping(chat_id)
    pending.setdefault(chat_id, []).append(item)
    old = tasks.get(chat_id)
    if old:
        old.cancel()
    tasks[chat_id] = asyncio.create_task(flush(chat_id))


async def flush(chat_id):
    try:
        await asyncio.sleep(WAIT_SECONDS)
    except asyncio.CancelledError:
        return
    tasks.pop(chat_id, None)
    items = pending.pop(chat_id, [])
    texts = [i[1] for i in items if i[0] == "text"]
    texts += [i[2] for i in items if i[0] == "img" and i[2]]
    imgs = [i[1] for i in items if i[0] == "img"]
    try:
        await bot.send_chat_action(chat_id, "typing")
        answer, photos = await ask(chat_id, "\n".join(texts), imgs or None)
        await send_parts(chat_id, answer, photos)
    except Exception as e:
        print("Ошибка:", e)
        await bot.send_message(chat_id, "Ой, сбой, напиши ещё раз")


# ---------------- ХЕНДЛЕРЫ ----------------
@dp.message(F.text)
async def on_text(m: Message):
    await enqueue(m.chat.id, ("text", m.text))


@dp.message(F.photo)
async def on_photo(m: Message):
    buf = io.BytesIO()
    await bot.download(m.photo[-1], destination=buf)
    await enqueue(m.chat.id, ("img", buf.getvalue(), m.caption or ""))


@dp.message(F.video | F.video_note)
async def on_video(m: Message):
    v = m.video or m.video_note
    if not v.thumbnail:
        await m.answer("Видос не смог открыть, скинь фоткой кадр")
        return
    buf = io.BytesIO()
    await bot.download(v.thumbnail, destination=buf)
    cap = (m.caption or "") + " (это кадр из видео, которое я скинул)"
    await enqueue(m.chat.id, ("img", buf.getvalue(), cap))


@dp.message(F.voice | F.audio)
async def on_voice(m: Message):
    f = m.voice or m.audio
    buf = io.BytesIO()
    await bot.download(f, destination=buf)
    try:
        r = await ai.audio.transcriptions.create(
            model=STT, file=("voice.ogg", buf.getvalue()), language="ru")
        await enqueue(m.chat.id, ("text", r.text))
    except Exception as e:
        print("Ошибка голоса:", e)
        await m.answer("Не разобрал голосовое, повтори")


@dp.message(F.location)
async def on_location(m: Message):
    user_loc[m.chat.id] = (m.location.latitude, m.location.longitude)
    await enqueue(m.chat.id, ("text", "(я скинул тебе свою геолокацию)"))


# ---------------- САМ ПИШЕТ ПЕРВЫМ ----------------
async def proactive():
    while True:
        await asyncio.sleep(300)
        hour = datetime.datetime.now().hour
        if hour < 9 or hour >= 23:      # ночью не пишет
            continue
        now = time.time()
        for cid in list(chats):
            if now < next_ping.setdefault(cid, now + random.randint(3600, 4 * 3600)):
                continue
            schedule_ping(cid, 3 * 3600, 7 * 3600)
            try:
                a, ph = await ask(
                    cid, "(системно: сам напиши мне первым, коротко, по-дружески, как будто "
                         "вспомнил обо мне или хочешь что-то обсудить. Не упоминай это сообщение.)",
                    hidden=True)
                await send_parts(cid, a, ph)
            except Exception as e:
                print("Ошибка proactive:", e)


async def main():
    asyncio.create_task(proactive())
    await dp.start_polling(bot)


asyncio.run(main())