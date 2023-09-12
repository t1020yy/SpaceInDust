import os

from openpyxl import Workbook


def save_tracks_to_excel(experiment, tracks):
    wb = Workbook()
    ws = wb.active

    ws.append(['Дата',
               'Напряжение, В',
               'Радиус частиц, м',
               ])

    ws.append([experiment['date'],
               experiment['voltage'],
               experiment['particles_radius']
               ])

    ws.append([''])

    ws.append(['#',
               'Рассчитанное смещение, пкс',
               'Длина трека, мм',
               'a',
               'b',
               'c',
               'Vy, м/с',
               'Vx, м/с',
               'V, м/с',
               'Угол взлета, градусов',
               'Радиус частицы, м',
               'Масса частицы, кг',
               'Заряд частицы, Кл'
               ])

    for i, track in enumerate(tracks):
        try:
            if track.parameters[8]:
                ws.append(
                [i,
                track.component1.shift[0],
                track.length,
                track.parabola[0],
                track.parabola[1],
                track.parabola[2],
                track.parameters[1],
                track.parameters[2],
                track.parameters[3],
                track.parameters[4],
                track.parameters[5],
                track.parameters[6],
                track.parameters[7]
                ])
            else:
                ws.append(
                [i,
                track.component1.shift[0],
                track.length,
                track.parabola[0],
                track.parabola[1],
                track.parabola[2],
                'Ошибка'
                ])
        except:
            pass

    wb.save("results.xlsx")

    os.system(f'start results.xlsx')