from aiogram import Bot, executor
from aiogram.dispatcher import Dispatcher
import shutil

import os

bot = Bot(token="5677890481:AAE3x49Uqngy4lvpGRB7BK9kvj-Rp5ig74o")
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def command_start(message):
    await bot.send_message(message.from_user.id, "Hello")


@dp.message_handler(content_types=['photo'])
async def photo_handler(message):

    path_for_download = 'telegram_fotos'
    await message.photo[-1].download(path_for_download+'/photo{}.png'.format(0))
    await bot.send_message(message.from_user.id, 'Processing photo...')
    os.system('python detect.py --weights last.pt --img 640 --conf 0.25 --source '+path_for_download)

    path_for_detected = 'runs/detect/'+[i for i in os.listdir('runs/detect') if 'exp' in i][0]
    for detected_photo in os.listdir(path_for_detected):
        photo = open(path_for_detected+'/'+detected_photo, "rb")
        await bot.send_photo(message.from_user.id, photo)
    photos = os.listdir('telegram_fotos')
    exps = os.listdir('runs/detect')

    for exp in exps:
        if os.path.isdir('runs/detect/'+exp):
            shutil.rmtree('runs/detect/' + exp)

    for photo in photos:
        os.remove('telegram_fotos/' + photo)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
