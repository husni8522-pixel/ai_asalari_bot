import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from langdetect import detect
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    filters
)
from openai import OpenAI
from docx import Document
from pypdf import PdfReader
from datetime import datetime

# ================== CONFIG ==================
DATA_DIR = "data"
INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"

CHUNK_SIZE = 1000
BATCH_SIZE = 32
TOP_K = 8
MAX_MEMORY = 5

# ================== LOAD ENV ==================
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))

client = OpenAI(api_key=OPENAI_KEY)

# ================== MEMORY & LOG ==================
user_memory = {}      # user_id -> savollar
questions_log = []    # savollar logi
user_stats = set()    # user_id lar
chat_log = {}         # chat_id -> {"title": str, "type": str}

# ================== LANGUAGE ==================
def detect_lang(text):
    try:
        l = detect(text)
        if l.startswith("ru"):
            return "ru"
        if l.startswith("en"):
            return "en"
        return "uz"
    except:
        return "uz"

# ================== BASIC CHAT ==================
def basic_chat(text):
    t = text.lower()
    
    # --- SALOM / HAYRLASHUV ---
    if any(w in t for w in ["salom", "assalomu", "hello", "hi", "привет"]):
        return {
            "uz": "Assalomu alaykum 😊 nima xizmat. savol bormi?",
            "ru": "Здравствуйте 😊 Задайте вопрос.",
            "en": "Hello 😊 Ask your question."
        }
    if any(w in t for w in ["xayr", "hayr", "goodbye", "bye", "пока", "до свидания"]):
        return {
            "uz": "Xayr! Sizni kutib qolamiz 😊",
            "ru": "До свидания! Будем рады вас видеть снова 😊",
            "en": "Goodbye! We hope to see you again 😊"
        }
    
    # --- RAHMAT / MINNATDORLIK ---
    if any(w in t for w in ["rahmat", "raxmat", "рахмат", "спасибо", "thank you"]):
        return {
            "uz": "Siz uchun hursandman. Arzimaydi 😊",
            "ru": "Рад помочь! Не за что 😊",
            "en": "I’m happy to help. You’re welcome 😊"
        }
    
    # --- BOTNING YARATUVCHISI / ALOQA ---
    if any(w in t for w in ["sani kim yaratgan", "seni kim yaratgan", "sani kim tuzgan", "seni kim tuzgan", "sani hujayining kim", 
    "seni hujayining kim", "sen kim", "sani kim", "kim san", "kim sen", "kim tuzgan"]):
        return {
            "uz": "Men akajonim Husniddin Zaripov tomonidan yaratilgan botman. Akamni duolarizda eslab qo‘ying @zhn8522😊",
            "ru": "Мен создан ботом Хусниддин Зарипов. Помните моего брата в ваших молитвах @zhn8522 😊",
            "en": "I am a bot created by Husniddin Zaripov. Keep my brother in your prayers @zhn8522 😊"
        }
    if any(w in t for w in ["qanday aloqaga chiqamiz", "aloqa", "contact", "how to contact"]):
        return {
            "uz": "Aloqa uchun: +998973850026 📞",
            "ru": "Связь: +998973850026 📞",
            "en": "Contact: +998973850026 📞"
        }
    
    return None

# ================== ASALARI ==================
ASALARI_WORDS = [
   "ari","asalari ich ketishi","asalarim","qishki ozuqa","arilar","asal","asalarichilik","asalarichi","ari oilasi","qirolicha",
"ona ari","ishchi ari","erkak ari","qandi","kandi","nuklius","asalarilarim",
"ari","асалари","bee","пчела","qishki oziqa","oziqa","asalari ozuqasi kamayib qolibdi qishda nima qilishim kerak",
"asalari","асалари","honeybee","медоносная пчела","Asalarilarning kuchi kam nima qilish kerak",
"asalarichilik","асаларичилик","beekeeping","пчеловодство",
"asalarichi","асаларичи","beekeeper","пчеловод",
"ari oilasi","ари оиласи","bee colony","пчелиная семья",
"ari koloniyasi","ари колонияси","bee colony","пчелиная колония",
"ona ari","она ари","queen bee","матка",
"qirolicha ari","қиролича ари","queen bee","пчелиная матка",
"ishchi ari","ишчи ари","worker bee","рабочая пчела",
"erkak ari","эркак ари","drone bee","трутень",
"truten","трутен","drone","трутень",
"matka","матка","queen","матка",
"ari uyasi","ари уяси","hive","улей",
"katta uya","катта уя","large hive","большой улей",
"kichik uya","кичик уя","small hive","малый улей",
"kop qavatli uya","кўп қаватли уя","multi hive","многоярусный улей",
"nuklius","нуклиус","nucleus hive","нуклеус",
"ramka","рамка","frame","рамка",
"asal ramkasi","асал рамкаси","honey frame","медовая рамка",
"bola ramkasi","бола рамкаси","brood frame","расплодная рамка",
"katak","катак","cell","ячейка",
"sota","сота","honeycomb","соты",
"mum","мум","wax","воск",
"mumli asos","мумли асос","wax foundation","вощина",
"panjara","панжара","queen excluder","разделительная решетка",
"asal","асал","honey","мёд",
"gul asali","гул асали","flower honey","цветочный мёд",
"tog asali","тоғ асали","mountain honey","горный мёд",
"perga","перга","bee bread","перга",
"gulchang","гулчанг","pollen","пыльца",
"propolis","прополис","propolis","прополис",
"qirollik suti","қироллик сути","royal jelly","маточное молочко",
"ari zahri","ари заҳри","bee venom","пчелиный яд",
"asal ajratgich","асал ажратгич","honey extractor","медогонка",
"medogonka","медогонка","honey extractor","медогонка",
"asal pichogi","асал пичоғи","uncapping knife","нож для распечатывания",
"tutatuvchi","тутатувчи","smoker","дымарь",
"dimar","димар","smoker","дымарь",
"ari kiyimi","ари кийими","bee suit","костюм пчеловода",
"niqob","ниқоб","veil","сетка",
"qolqop","қўлқоп","gloves","перчатки",
"oziqlantirish","озиқлантириш","feeding","кормление",
"shakar","шакар","sugar","сахар",
"sirop","сироп","syrup","сироп",
"kandi","канди","candy feed","канди",
"qishki ozuqa","қишки озуқа","winter feed","зимний корм",
"bahorgi oziqlantirish","баҳорги озиқлантириш","spring feeding","весеннее кормление",
"kuzgi oziqlantirish","кузги озиқлантириш","autumn feeding","осеннее кормление",
"varroa","варроа","varroa mite","клещ варроа",
"nosema","нозема","nosema","нозематоз",
"akarapidoz","акарапидоз","acarapidosis","акарапидоз",
"amerikan chirishi","американ чириши","american foulbrood","американский гнилец",
"yevropa chirishi","европа чириши","european foulbrood","европейский гнилец",
"davolash","даволаш","treatment","лечение",
"profilaktika","профилактика","prevention","профилактика",
"oksalat kislota","оксалат кислотаси","oxalic acid","щавелевая кислота",
"formik kislota","формик кислотаси","formic acid","муравьиная кислота",
"timol","тимол","thymol","тимол",
"asal yigimi","асал йиғими","honey harvest","сбор мёда",
"asal oqimi","асал оқими","honey flow","медосбор",
"nektar","нектар","nectar","нектар",
"nektar yigish","нектар йиғиш","nectar collection","сбор нектара",
"ari uchishi","ари учиши","bee flight","лёт пчёл",
"swarmlash","свармлаш","swarming","роение",
"roy olish","рой олиш","swarm capture","отлов роя",
"ari kundaligi","ари кундалиги","beekeeper journal","журнал пчеловода",
"uya tekshiruvi","уя текшируви","hive inspection","осмотр улья",
"ari salomatligi","ари саломатлиги","bee health","здоровье пчёл",
"mum qurti","мум қурти","wax moth","восковая моль",
"ari zotlari","ари зотлари","bee breeds","породы пчёл",
"italyan ari","italyan zoti","италян ари","italian bee","итальянская пчела",
"karnika ari","karnika zoti","карника ари","carnica bee","карника",
"kavkaz ari","kavkaz zoti","кавказ ари","caucasian bee","кавказская пчела",
"karpat ari","karpat zoti","карпат ари","carpathian bee","карпатская пчела",
"rus ari","rus zoti","рус ари","russian bee","русская пчела",
"orta yevropalik ari","ўрта европалик ари","central european bee","среднеевропейская пчела",
"tog ari","тоғ ари","mountain bee","горная пчела",
"yovvoyi ari","ёввойи ари","wild bee","дикая пчела",
"medonos ari","медонос ари","honey bee","медоносная пчела",
"qora ari","қора ари","black bee","чёрная пчела",

"ari rivojlanishi","ари ривожланиши","bee development","развитие пчелы",
"tuxum","тухум","egg","яйцо",
"lichinka","личинка","larva","личинка",
"gumbak","ғумбак","pupa","куколка",
"bola ari","бола ари","brood","расплод",
"ochiq bola","очиқ бола","open brood","открытый расплод",
"yopiq bola","ёпиқ бола","sealed brood","печатный расплод",

"ona ari yetishtirish","она ари етиштириш","queen rearing","вывод маток",
"ona ari belgilash","она ари белгилаш","queen marking","метка матки",
"ona ari almashtirish","она ари алмаштириш","queen replacement","замена матки",
"ona ari qabul qilish","она ари қабул қилиш","queen introduction","подсадка матки",
"ona ari qafasi","она ари қафаси","queen cage","клеточка для матки",

"ari bolinishi","ари бўлиниши","colony splitting","деление семьи",
"suniy bolish","сунъий бўлиш","artificial split","искусственное деление",
"ari kuchaytirish","ари кучайтириш","colony boosting","усиление семьи",

"ari xulqi","ари хулқи","bee behavior","поведение пчёл",
"swarmlash holati","свармлаш ҳолати","swarm behavior","роевое состояние",
"ari tajovuzi","ари тажовузи","bee aggression","агрессивность пчёл",
"ari tinchligi","ари тинчлиги","bee calmness","миролюбие",

"asal saqlash","асал сақлаш","honey storage","хранение мёда",
"mum saqlash","мум сақлаш","wax storage","хранение воска",
"perga saqlash","перга сақлаш","perga storage","хранение перги",
"asal idishi","асал идиши","honey container","тара для мёда",

"asal qirqish","асал қирқиш","uncapping","распечатывание",
"mum eritish","мум эритиш","wax melting","топка воска",
"mum tozalash","мум тозалаш","wax cleaning","очистка воска",

"ari tekshiruvi","ари текшируви","bee inspection","осмотр пчёл",
"ramka aylantirish","рамка айлантириш","frame rotation","ротация рамок",
"uya tozalash","уя тозалаш","hive cleaning","чистка улья",
"uya dezinfeksiya","уя дезинфекция","hive disinfection","дезинфекция улья",

"ari kasalliklari","ари касалликлари","bee diseases","болезни пчёл",
"virus kasalligi","вирус касаллиги","viral disease","вирусное заболевание",
"zamburug kasalligi","замбуруғ касаллиги","fungal disease","грибковое заболевание",

"mum qurti lichinkasi","мум қурти личинкаси","wax moth larva","личинка восковой моли",
"mum qurti davolash","мум қурти даволаш","wax moth treatment","лечение восковой моли",

"ari ozuqasi","ари озуқаси","bee feed","корм для пчёл",
"protein ozuqa","протеин озуқа","protein feed","белковый корм",
"gulchangli ozuqa","гулчангли озуқа","pollen feed","пыльцевой корм",

"qishga tayyorlash","қишга тайёрлаш","winter preparation","подготовка к зиме",
"qishlash","қишлаш","wintering","зимовка",
"ari qishlashi","ари қишлаши","bee wintering","зимовка пчёл",
"uyani isitish","уяни иситиш","hive insulation","утепление улья",

"yozgi parvarish","ёзги парвариш","summer care","летний уход",
"bahorgi parvarish","баҳорги парвариш","spring care","весенний уход",
"kuzgi parvarish","кузги парвариш","autumn care","осенний уход",

"apiari joylashuvi","апиари жойлашуви","apiary layout","расположение пасеки",
"uyalar oraligi","уялир оралиғи","hive spacing","расстояние между ульями",
"apiari xavfsizligi","апиари хавфсизлиги","apiary security","безопасность пасеки",

"asal hosildorligi","асал ҳосилдорлиги","honey yield","урожай мёда",
"ari samaradorligi","ари самарадорлиги","bee productivity","продуктивность пчёл",
"koloniya holati","колония ҳолати","colony condition","состояние семьи",
"asal ishlab chiqarish","асал ишлаб чиқариш","honey production","производство мёда",
"mum ishlab chiqarish","мум ишлаб чиқариш","wax production","производство воска",
"propolis ishlab chiqarish","прополис ишлаб чиқариш","propolis production","производство прополиса",
"perga ishlab chiqarish","перга ишлаб чиқариш","perga production","производство перги",

"asal tahlili","асал таҳлили","honey analysis","анализ мёда",
"gulchang tahlili","гулчанг таҳлили","pollen analysis","анализ пыльцы",
"asal sifati","асал сифати","honey quality","качество мёда",
"soxta asal","сохта асал","fake honey","поддельный мёд",

"ari zahri yigish","ари заҳри йиғиш","bee venom collection","сбор пчелиного яда",
"ari zahri ajratish","ари заҳри ажратиш","bee venom extraction","извлечение пчелиного яда",
"qirollik suti yigish","қироллик сути йиғиш","royal jelly harvesting","сбор маточного молочка",

"asal ramkasini chiqarish","асал рамкасини чиқариш","frame removal","извлечение рамки",
"asal suzish","асал сузиш","honey filtering","фильтрация мёда",
"asal quyish","асал қуйиш","honey bottling","разлив мёда",

"ari uchish masofasi","ари учиш масофаси","foraging range","радиус лёта",
"ari yem izlash","ари ем излаш","foraging behavior","кормодобывание",
"ari yo‘nalishi","ари йўналиши","bee orientation","ориентация пчёл",

"ari genetikasi","ари генетикаси","bee genetics","генетика пчёл",
"ona ari tanlash","она ари танлаш","queen selection","отбор маток",
"suniy uruglantirish","сунъий уруғлантириш","artificial insemination","искусственное осеменение",

"erkak ari boshqaruvi","эркак ари бошқаруви","drone management","управление трутнями",
"erkak ari ko‘payishi","эркак ари кўпайиши","drone production","вывод трутней",

"ari joylashuvi","ари жойлашуви","bee positioning","расположение пчёл",
"ari hududi","ари ҳудуди","bee territory","территория пчёл",

"ari stressi","ари стресси","bee stress","стресс пчёл",
"ari shovqini","ари шовқини","bee noise","шум пчёл",
"ari signallari","ари сигналлари","bee signals","сигналы пчёл",

"ari nafasi","ари нафаси","bee respiration","дыхание пчёл",
"ari harorati","ари ҳарорати","bee temperature","температура пчёл",

"uya shamollatish","уя шамоллатиш","hive ventilation","вентиляция улья",
"uya namligi","уя намлиги","hive humidity","влажность улья",
"uya harorati","уя ҳарорати","hive temperature","температура улья",

"ari paraziti","ари паразити","bee parasite","паразиты пчёл",
"kanalar","каналар","mites","клещи",
"ari biti","ари бити","bee louse","пчелиная вошь",

"organik davolash","органик даволаш","organic treatment","органическое лечение",
"kimyoviy davolash","кимёвий даволаш","chemical treatment","химическое лечение",
"dori dozalash","дори дозалаш","drug dosage","дозировка препарата",

"ari salomatlik tekshiruvi","ари саломатлик текшируви","bee health check","проверка здоровья пчёл",
"kasallikni oldini olish","касалликни олдини олиш","disease prevention","профилактика заболеваний",

"ari kundalik","ари кундалик","bee journal","дневник пасеки",
"apiari xaritasi","апиари харитаси","apiary mapping","карта пасеки",
"uyani raqamlash","уяни рақамлаш","hive numbering","нумерация ульев",
"uyani belgilash","уяни белгилаш","hive labeling","маркировка ульев",

"asal bozori","асал бозори","honey market","рынок мёда",
"asal savdosi","асал савдоси","honey trade","торговля мёдом",
"asal narxi","асал нархи","honey price","цена мёда",

"ari changlatishi","ари чанглатиши","pollination","опыление",
"ekin changlatish","экин чанглатиш","crop pollination","опыление культур",
"bog‘ changlatish","боғ чанглатиш","garden pollination","опыление сада",

"ari tashish","ари ташиш","bee transportation","перевозка пчёл",
"ko‘chma apiari","кўчма апиари","migratory beekeeping","кочевое пчеловодство",

"ari himoyasi","ари ҳимояси","bee protection","защита пчёл",
"yirtqichlardan himoya","йиртқичлардан ҳимоя","predator protection","защита от хищников",
"ari instinkti","ари инстинкти","bee instinct","инстинкт пчёл",
"ari xotirasi","ари хотираси","bee memory","память пчёл",
"ari hid sezishi","ари ҳид сезиши","bee smell sense","обоняние пчёл",
"ari korishi","ари кўриши","bee vision","зрение пчёл",

"ari uchish tezligi","ари учиш тезлиги","bee flight speed","скорость полёта пчёл",
"ari uchish balandligi","ари учиш баландлиги","bee flight height","высота лёта пчёл",
"ari energiyasi","ари энергияси","bee energy","энергия пчёл",

"ari ish faoliyati","ари иш фаолияти","bee activity","активность пчёл",
"ari ish vaqti","ари иш вақти","bee working time","рабочее время пчёл",
"ari dam olishi","ари дам олиши","bee rest","отдых пчёл",

"ari himoya signali","ари ҳимоя сигнали","defense signal","сигнал защиты",
"ari hujum holati","ари ҳужум ҳолати","attack behavior","агрессивное поведение",
"ari chaqishi","ари чақиши","bee sting","укус пчелы",
"ari chaqishi ogriq","ари чақиши оғриқ","bee sting pain","боль от укуса",

"ari zahar bezlari","ари заҳар безлари","venom glands","ядовитые железы",
"ari zahar miqdori","ари заҳар миқдори","venom amount","количество яда",

"ari aloqa raqsi","ари алоқа рақси","waggle dance","танец пчёл",
"ari signal raqsi","ари сигнал рақси","signal dance","сигнальный танец",
"ari yonalish raqsi","ари йўналиш рақси","orientation dance","ориентирующий танец",

"ari ozuqa manbai","ари озуқа манбаи","food source","источник корма",
"ari gul tanlashi","ари гул танлаши","flower selection","выбор цветка",
"ari rang ajratishi","ари ранг ажратиши","color perception","различие цветов",

"ari uyaga qaytishi","ари уяга қайтиши","homing behavior","возвращение в улей",
"ari yol topishi","ари йўл топиши","navigation","навигация пчёл",

"ari ekologiyasi","ари экологияси","bee ecology","экология пчёл",
"ari muhitga moslashuvi","ари муҳитга мослашуви","adaptation","адаптация пчёл",
"ari iqlimga moslashuvi","ари иқлимга мослашуви","climate adaptation","адаптация к климату",

"ari populyatsiyasi","ари популяцияси","bee population","популяция пчёл",
"ari soni kamayishi","ари сони камайиши","bee decline","сокращение пчёл",
"ari yoqolishi","ари йўқолиши","bee loss","гибель пчёл",

"ari zaharlanishi","ари заҳарланиши","bee poisoning","отравление пчёл",
"pestitsid ta'siri","пестицид таъсири","pesticide impact","влияние пестицидов",
"kimyoviy moddalar","кимёвий моддалар","chemicals","химические вещества",

"ari himoya qonuni","ари ҳимоя қонуни","bee protection law","закон о защите пчёл",
"ari muhofazasi","ари муҳофазаси","bee conservation","охрана пчёл",

"ari va qishloq xojaligi","ари ва қишлоқ хўжалиги","bees and agriculture","пчёлы и сельское хозяйство",
"ari va ekinlar","ари ва экинлар","bees and crops","пчёлы и культуры",
"ari va tabiat","ари ва табиат","bees and nature","пчёлы и природа",

"ari mahsuldorligi","ари маҳсулдорлиги","bee efficiency","эффективность пчёл",
"ari iqtisodiy foyda","ари иқтисодий фойда","economic value","экономическая ценность пчёл",

"ari seleksiyasi","ари селекцияси","bee breeding","селекция пчёл",
"ari naslchilik","ари наслчилик","bee breeding","разведение пчёл",
"ari zoti yaxshilash","ари зоти яхшилаш","breed improvement","улучшение породы",

"ari oquv mashgulot","ари ўқув машғулот","beekeeping training","обучение пчеловодству",
"asalarichilik kursi","асаларичилик курси","beekeeping course","курс пчеловодства",
"asalarichilik kitobi","асаларичилик китоби","beekeeping book","книга по пчеловодству",

"ari texnologiyasi","ари технологияси","beekeeping technology","технологии пчеловодства",
"zamonaviy asalarichilik","замонавий асаларичилик","modern beekeeping","современное пчеловодство",
"an'anaviy asalarichilik","анъанавий асаларичилик","traditional beekeeping","традиционное пчеловодство",

"ari statistikasi","ари статистикаси","bee statistics","статистика пчёл",
"ari ilmiy tadqiqot","ари илмий тадқиқот","scientific research","научные исследования пчёл",
"ari monitoringi","ари мониторинги","bee monitoring","мониторинг пчёл",

"ari himoya loyihasi","ари ҳимоя лойиҳаси","bee protection project","проект защиты пчёл",
"ari ekologik loyiha","ари экологик лойиҳа","ecological project","экологический проект",

"ari va iqlim ozgarishi","ари ва иқлим ўзгариши","climate change impact","влияние изменения климата",
"ari global muammo","ари глобал муаммо","global issue","глобальная проблема",
"asalariy","асаларий","bee","пчела",
"asalri","асалри","honeybee","медоносная пчела",
"asalarichik","асаларичик","beekeeper","пчеловод",
"ari oylasi","ари ойлоси","bee colony","пчелиная семья",
"onari","онари","queen bee","матка",
"qirolichaari","қироличаари","queen bee","пчелиная матка",
"ishchi ari","ишчи ари","worker bee","рабочая пчела",
"erkak ari","эркакар","drone bee","трутень",
"trutten","труттен","drone","трутень",
"matka","матка","queen","матка",
"ari uyasi","ари уяси","hive","улей",
"katta uya","катта уя","large hive","большой улей",
"kichik uya","кичик уя","small hive","малый улей",
"kopqavatli uya","кўпқаватли уя","multi hive","многоярусный улей",
"ramkaa","рамкаа","frame","рамка",
"asarramka","асаррамка","honey frame","медовая рамка",
"bola katak","бола катак","brood frame","расплодная рамка",
"katak","катак","cell","ячейка",
"sotaa","сотаа","honeycomb","соты",
"mumli asosss","мумли асоссс","wax foundation","вощина",
"panjara","панжара","queen excluder","разделительная решетка",
"asal","асал","honey","мёд",
"gul asal","гул асал","flower honey","цветочный мёд",
"tog asal","тоғасал","mountain honey","горный мёд",
"perga","перга","bee bread","перга",
"gulchang","гулчанга","pollen","пыльца",
"propolis","прополис","propolis","прополис",
"qirollik suti","қироллик сути","royal jelly","маточное молочко",
"ari zahri","ари заҳри","bee venom","пчелиный яд",
"asal ajrttgich","асал ажртгич","honey extractor","медогонка",
"medogonka","медогонка","honey extractor","медогонка",
"asal pichogi","асал пичогги","uncapping knife","нож для распечатывания",
"tutatuvchi","тутатувчи","smoker","дымарь",
"dimar","димар","smoker","дымарь",
"ari kiyimi","ари кийими","bee suit","костюм пчеловода",
"niqqob","ниққоб","veil","сетка",
"qolqqop","қолққоп","gloves","перчатки",
"oziqllantirish","озиқлллантириш","feeding","кормление",
"shakkar","шаккар","sugar","сахар",
"siropp","сиропп","syrup","сироп",
"kandii","канди","candy feed","канди",
"qishkooza","қишкооза","winter feed","зимний корм",
"bahor oziqa","баҳор озиқа","spring feeding","весеннее кормление",
"kuzgii oziq","кузгии озиқ","autumn feeding","осеннее кормление",
"varoaa","вароаа","varroa mite","клещ варроа",
"nosemma","ноземма","nosema","нозематоз",
"akarapidozz","акарапидозз","acarapidosis","акарапидоз",
"amerikaan chirishi","америкаан чириши","american foulbrood","американский гнилец",
"yevropaa chirishi","европаа чириши","european foulbrood","европейский гнилец",
"varroa nima","варроа нима","what is varroa","что такое варроа",
"ona ari yo‘q","она ари йўқ","queen bee missing","нет матки",
"asal qanday olinadi","асал қандай олинади","how is honey collected","как собирают мёд",
"ari zahri qanday yig‘iladi","ари заҳри қандай йиғилади","how to collect bee venom","как собирают пчелиный яд",
"perga nima","перга нима","what is perga","что такое перга",
"qirollik suti nima","қироллик сути нима","what is royal jelly","что такое маточное молочко",
"mum nima","мум нима","what is wax","что такое воск",
"propolis nima","прополис нима","what is propolis","что такое прополис",
"ari qanday oziqlanadi","ари қандай озиқланади","how to feed bees","как кормить пчёл",
"qishki oziqa qanday beriladi","қишки озиқ қандай берилади","how to feed in winter","как кормить зимой",
"bahorgi oziqa berish","баҳорги озиқ бериш","spring feeding","весеннее кормление",
"ari kasalligi alomatlari","ари касаллиги аломатлари","bee disease symptoms","симптомы болезни пчёл",
"varroa belgilari","варроа белгилари","varroa symptoms","симптомы варроа",
"nosema alomatlari","нозема аломатлари","nosema symptoms","симптомы нозема",
"amerikan chirish belgilari","американ чириш белгилари","american foulbrood symptoms","симптомы американского гнильца",
"yevropa chirish belgilari","европа чириш белгилари","european foulbrood symptoms","симптомы европейского гнильца",
"mum qurti qanday oldini olish","мум қурти қандай олдини олиш","how to prevent wax moth","как предотвратить восковую моль",
"ari bolinishi qanday","ари бўлиниши қандай","how to split a colony","как делить семью",
"ona ari qanday tanlanadi","она ари қандай танланади","how to select queen","как выбрать матку",
"ona ari qanday belgilanadi","она ари қандай белгиланади","how to mark queen","как отметить матку",
"ari swarm qilsa nima qilish kerak","ари сварм қилса нима қилиш керак","what to do if bees swarm","что делать если рой улей",
"uyani qanday tozalash","уяни қандай тозалаш","how to clean hive","как чистить улей",
"ari qancha kun ishlaydi","ари қанча кун ишлайди","how long do worker bees live","сколько живут рабочие пчёлы",
"ari qancha asal beradi","ари қанча асал беради","how much honey bees produce","сколько мёда даёт пчела",
"ari uchish masofasi","ари учиш масофаси","bee foraging distance","дальность облёта пчёл",
"ari qanday changlatadi","ари қандай чанглатади","how bees pollinate","как пчёлы опыляют",
"ari qishda qanday yashaydi","ари қишда қандай яшайди","how bees survive winter","как пчёлы переживают зиму",
"ari kasalliklaridan qanday himoya qilish","ари касалликларидан қандай ҳимоя қилиш","how to prevent bee diseases","как защитить пчёл от болезней",
"medogonka qanday ishlaydi","медогонка қандай ишлайди","how honey extractor works","как работает медогонка",
"ona ari qafasini qanday ishlatish","она ари қафасини қандай ишлатиш","how to use queen cage","как использовать клетку для матки",
"ari shamollatishni qanday qilish","ари шамоллатишни қандай қилиш","how to ventilate hive","как проветривать улей",
"ari namlikni qanday nazorat qilish","ари намликни қандай назорат қилиш","how to control hive humidity","как контролировать влажность улья",
"varroa","варроа","varroa mite","клещ варроа",
"varroa davolash","варроа даволаш","varroa treatment","лечение варроа",
"formik kislota","формик кислота","formic acid","муравьиная кислота",
"oksalat kislota","оксалат кислота","oxalic acid","оксаловая кислота",
"timol","тимол","thymol","тимол",
"nosema","нозема","nosema","нозематоз",
"nosema davolash","нозема даволаш","nosema treatment","лечение ноземы",
"amerikan chirishi","американ чириш","american foulbrood","американский гнилец",
"amerikan chirishi davolash","американ чириш даволаш","american foulbrood treatment","лечение американского гнильца",
"yevropa chirishi","европа чириш","european foulbrood","европейский гнилец",
"yevropa chirishi davolash","европа чириш даволаш","european foulbrood treatment","лечение европейского гнильца",
"akarapidoz","акарапидоз","acarapidosis","акарапидоз",
"akarapidoz davolash","акарапидоз даволаш","acarapidosis treatment","лечение акарапидоза",
"mum qurti","мум қурти","wax moth","восковая моль",
"mum qurti davolash","мум қурти даволаш","wax moth treatment","лечение восковой моли",
"ari zahri ortiqcha","ари заҳри ортиқча","bee venom overdose","передозировка пчелиного яда",
"ari zahri tekshirish","ари заҳри текшириш","bee venom check","проверка пчелиного яда",
"ari kasalligi","ари касаллиги","bee disease","болезнь пчёл",
"kasallikni oldini olish","касалликни олдини олиш","disease prevention","профилактика заболеваний",
"dori berish","дори бериш","medication","медикамент",
"organik dori","органик дори","organic medicine","органическое лечение",
"kimyoviy dori","кимёвий дори","chemical medicine","химическое лечение",
"ari himoya dorisi","ари ҳимоя дориси","bee protection medicine","лекарство для пчёл",
"profilaktika","профилактика","preventive measures","профилактические меры",
"ari sog‘lomligi","ари соғломлиги","bee health","здоровье пчёл",
"virusli kasallik","вирусли касаллик","viral disease","вирусная болезнь",
"zamburug‘li kasallik","замбуруғли касаллик","fungal disease","грибковое заболевание",
"parazitlar","паразитлар","parasites","паразиты",
"ari kasalliklari belgilari","ари касалликлари белгилари","bee disease symptoms","симптомы болезней пчёл",
"ari dori tavsiyalari","ари дори тавсиялари","bee medicine recommendations","рекомендации по лекарствам для пчёл",
"varroa profilaktikasi","варроа профилактикаси","varroa prevention","профилактика варроа",
"nosema profilaktikasi","нозема профилактикаси","nosema prevention","профилактика ноземы",
"amerikan chirish profilaktikasi","американ чириш профилактикаси","american foulbrood prevention","профилактика американского гнильца",
"yevropa chirish profilaktikasi","европа чириш профилактикаси","european foulbrood prevention","профилактика европейского гнильца",
"mum qurti profilaktikasi","мум қурти профилактикаси","wax moth prevention","профилактика восковой моли",
"davolash usullari","даволаш усуллари","treatment methods","методы лечения",
"ari sog‘lom turmushi","ари соғлом турмуши","healthy bee management","здоровый уход за пчелами",
"medogonka","медогонка","honey extractor","медогонка",
"asal pichog‘i","асал пичог‘и","uncapping knife","нож для распечатывания",
"tutatuvchi","тутатувчи","smoker","дымарь",
"dimar","димар","smoker","дымарь",
"ari kiyimi","ари кийими","bee suit","костюм пчеловода",
"qo‘lqop","қўлқоп","gloves","перчатки",
"niqob","ниқоб","veil","маска",
"ramka","рамка","frame","рамка",
"katak","катак","cell","ячейка",
"panjara","панжара","queen excluder","разделительная решетка",
"mumli asos","мумли асос","wax foundation","вощина",
"asali panjara","асали панжара","honey frame","медовая рамка",
"asal ajratgich","асал ажратгич","honey separator","разделитель мёда",
"honey gate","хани гейт","honey gate","кран для мёда",
"uncapping fork","анкаппинг форк","uncapping fork","вилка для распечатывания",
"bee brush","би браш","bee brush","кисть для пчёл",
"bee feeder","би фидер","bee feeder","кормушка для пчёл",
"swarm trap","сварм трап","swarm trap","ловушка для роя",
"swarm box","сварм бокс","swarm box","ящик для роя",
"nectar trap","нектар трап","nectar trap","ловушка для нектара",
"hive tool","хайв тул","hive tool","инструмент для улья",
"medogonka tozalash","медогонка тозалаш","extractor cleaning","чистка медогонки",
"gloves tozalash","гловес тозалаш","gloves cleaning","чистка перчаток",
"veil tozalash","вейл тозалаш","veil cleaning","чистка маски",
"frame rotation","рамка айлантириш","frame rotation","ротация рамок",
"wax foundation replacement","вощина алмаштириш","wax foundation replacement","замена вощины",
"queen cage","она ари қафаси","queen cage","клеточка для матки",
"bee suit maintenance","ари кийими тозалаш","bee suit maintenance","уход за костюмом",
"nectar collector","нектар коллектор","nectar collector","сборщик нектара",
"honey extractor parts","медогонка қисмлари","honey extractor parts","запчасти медогонки",
"bee smoker fuel","димар ёқилғи","smoker fuel","топливо для дымаря",
"hive numbering","уя рақамлаш","hive numbering","нумерация ульев",
"apiary mapping","апиари харитаси","apiary mapping","карта пасеки",
"hive labeling","уя белгилаш","hive labeling","маркировка ульев",
"inspection tools","текширув асбоблари","inspection tools","инструменты для осмотра",
"protective clothing","ҳимоя кийими","protective clothing","защитная одежда",
"bee feeder tank","ари озуқаси идиши","bee feeder tank","ёмкость для кормушки",
"perga storage container","перга сақлаш идиши","perga storage container","контейнер для перги",
"honey storage tank","асал сақлаш идиши","honey storage tank","ёмкость для мёда",
"wax storage container","мум сақлаш идиши","wax storage container","контейнер для воска",
"apiary security fence","апиари хавфсизлик тўсиғи","apiary security fence","ограждение пасеки",
"smoker nozzle","димар насоси","smoker nozzle","насадка для дымаря",
"bee marking pen","ари белгилаш қалами","bee marking pen","маркер для пчёл",
"nectar feeder","нектар озуқаси идиши","nectar feeder","кормушка для нектара",
"hive thermometer","уя термометри","hive thermometer","термометр для улья",
"hive hygrometer","уя гигрометри","hive hygrometer","гигрометр для улья",
"honey bottling equipment","асал қуйиш ускуналари","honey bottling equipment","оборудование для разлива мёда",
"frame grip","рамка ушлагич","frame grip","захват рамки",
"queen introduction tool","она ари қўйиш ускуналари","queen introduction tool","инструмент для подсадки матки",
"hive ventilator","уя вентилятори","hive ventilator","вентилятор улья",
"apiary layout tools","апиари жойлашув ускуналари","apiary layout tools","инструменты для планировки пасеки"

]

def is_asalari(text):
    return any(w in text.lower() for w in ASALARI_WORDS)

# ================== FILES ==================
def read_file(path):
    if path.endswith(".docx"):
        return "\n".join(p.text for p in Document(path).paragraphs)
    if path.endswith(".pdf"):
        return "\n".join(p.extract_text() for p in PdfReader(path).pages if p.extract_text())
    if path.endswith(".txt"):
        return open(path, encoding="utf-8", errors="ignore").read()
    return ""

def chunk_text(text):
    return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

# ================== INDEX ==================
# ================== INDEX ==================
def build_index():
    print("♻️ INDEX YARATILYAPTI...")
    docs = []

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for f in os.listdir(DATA_DIR):
        if f.endswith((".pdf", ".docx", ".txt")):
            text = read_file(os.path.join(DATA_DIR, f))
            for c in chunk_text(text):
                if len(c.strip()) > 50 and is_asalari(c):
                    docs.append(c.strip())

    if not docs:
        print("❌ DATA papkada mos hujjat yo‘q")
        return

    vectors = []
    for i in range(0, len(docs), BATCH_SIZE):
        r = client.embeddings.create(
            model="text-embedding-3-small",
            input=docs[i:i+BATCH_SIZE]
        )
        vectors.extend([d.embedding for d in r.data])

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, INDEX_FILE)
    pickle.dump(docs, open(META_FILE, "wb"))

    print("✅ INDEX TAYYOR")

def index_invalid():
    if not os.path.exists(INDEX_FILE):
        return True
    if not os.path.exists(META_FILE):
        return True
    if os.path.getsize(INDEX_FILE) < 1000:
        return True
    if os.path.getsize(META_FILE) < 50:
        return True
    return False

def search_docs(q):
    index = faiss.read_index(INDEX_FILE)
    texts = pickle.load(open(META_FILE, "rb"))
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[q]
    ).data[0].embedding

    _, I = index.search(np.array([emb]).astype("float32"), TOP_K)
    return [texts[i] for i in I[0]]

# ================== AI ANSWER ==================
def ai_answer(uid, q):
    lang = detect_lang(q)
    basic = basic_chat(q)
    if basic:
        return basic[lang]

    if uid not in user_memory:
        user_memory[uid] = []

    if not is_asalari(q):
        return {
            "uz": "🐝 Bu bot faqat asalarichilik uchun.",
            "ru": "🐝 Бот только для пчеловодства.",
            "en": "🐝 This bot is for beekeeping only."
        }[lang]

    user_memory[uid].append(q)

    ctx = "\n".join(search_docs(q))
    if not ctx:
        return {
            "uz": "❌ Ma’lumot topilmadi.",
            "ru": "❌ Информация не найдена.",
            "en": "❌ No information found."
        }[lang]

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": "You are an expert beekeeper."},
                  {"role": "user", "content": f"{ctx}\n\nSavol: {q}"}],
        temperature=0.3
    )
    return r.choices[0].message.content.strip()

# ================== BUTTON ==================
def reset_button():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Yangi savol", callback_data="reset")]])

# ================== LOG CHAT ==================
async def log_chat(update: Update):
    chat = update.effective_chat
    user_stats.add(update.effective_user.id)
    if chat.id not in chat_log:
        chat_log[chat.id] = {
            "title": chat.title or f"{update.effective_user.first_name}",
            "type": chat.type
        }

# ================== HANDLERS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await log_chat(update)
    await update.message.reply_text(
        "🐝 Asalarichilik AI botga xush kelibsiz!",
        reply_markup=reset_button()
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await log_chat(update)
    uid = update.effective_user.id
    q = update.message.text.strip()
    questions_log.append(q)
    ans = ai_answer(uid, q)

    await update.message.reply_text(ans, reply_markup=reset_button())

    if ADMIN_ID:
        await context.bot.send_message(
            ADMIN_ID,
            f"👤 USER ID: {uid}\n🕒 {datetime.now()}\n❓ Savol: {q}\n✅ Javob: {ans}\n"
            f"💬 Chat: {chat_log[update.effective_chat.id]['title']} ({chat_log[update.effective_chat.id]['type']})"
        )

async def reset_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    uid = query.from_user.id
    user_memory.pop(uid, None)
    await query.answer()
    await query.message.reply_text(
        "✅ Context tozalandi. Yangi savol berishingiz mumkin.",
        reply_markup=reset_button()
    )

# ================== ADMIN ==================
async def reindex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("❌ Sizda bu komandani ishlatish huquqi yo‘q.")
        return
    await update.message.reply_text("♻️ Indeks yangilanmoqda...")
    build_index()
    await update.message.reply_text("✅ Indeks tayyor")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("❌ Sizda bu komandani ishlatish huquqi yo‘q.")
        return
    chats = "\n".join([f"{v['title']} ({v['type']})" for v in chat_log.values()])
    await update.message.reply_text(
        f"📊 Foydalanuvchilar: {len(user_stats)}\n"
        f"📩 Savollar: {len(questions_log)}\n"
        f"💬 Guruhlar/kanallar:\n{chats}"
    )

# ================== MAIN ==================
if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        build_index()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reindex", reindex))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(reset_callback, pattern="^reset$"))

    print("🐝 BOT ISHGA TUSHDI")
    app.run_polling()
