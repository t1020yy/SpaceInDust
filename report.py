import io
import os

import PIL

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas 
from reportlab.pdfbase import pdfmetrics, ttfonts
from reportlab.platypus import SimpleDocTemplate, Image, Table, Paragraph, PageBreak, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from particle_track import Particle_track
from connected_component import Connected_component


def make_report(tracks):
    MyFontObject = ttfonts.TTFont('Times', 'times.ttf')
    pdfmetrics.registerFont(MyFontObject)

    sample_style_sheet = getSampleStyleSheet()
    head_style = sample_style_sheet['BodyText']
    head_style.fontSize = 14
    head_style.fontName = 'Times'
    body_style = sample_style_sheet['BodyText']
    body_style.fontSize = 12
    body_style.fontName = 'Times'

    doc = SimpleDocTemplate('report.pdf', pagesize=A4, 
                            topMargin=1.5*cm, leftMargin=2*cm,
                            rightMargin=2*cm, bottomMargin=1.5*cm)

    table_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('VALIGN', (0, 0), (-1, -1), 'CENTER')])

    flowables = []

    flowables.append(Paragraph(f'Распределение заряда частиц', head_style))
    flowables.append(Image(_draw_q_hist(tracks), width=12*cm, height=12*cm))
    flowables.append(Paragraph(f'3D траектории частиц', head_style))
    flowables.append(Image(_draw_tracks_3d(tracks), width=12*cm, height=12*cm))
    flowables.append(PageBreak())

    angle_values = []  # 存储角度值的列表

    for i, track in enumerate(tracks):
        image1 = PIL.Image.fromarray(track.component1.get_visualized_img())
        image2 = PIL.Image.fromarray(track.component2.get_visualized_img())
        
    
        flowables.append(Paragraph(f'Трек №{i}', head_style))

        flowables.append(Table([[_draw_image(image1, width=8*cm, height=8*cm), _draw_image(image2, width=8*cm, height=8*cm)],
                                 [Image(track.draw_3d_figure(), width=8*cm, height=8*cm), Image(track.draw_2d_figure(), width=8*cm, height=8*cm)]],
                         colWidths=[9*cm, 9*cm],
                         rowHeights=[9*cm, 9*cm], style=table_style))

        flowables.append(Paragraph(f'Максимум корреляционной функции {track.component1.correlation_max:.3f}', body_style))
        flowables.append(Paragraph(f'Рассчитанное смещение {track.component1.shift} пкс', body_style))
        flowables.append(Paragraph(f'Длина трека {track.length:.4f} мм', body_style))
        flowables.append(Paragraph(f'Параметры параболы a={track.parabola[0]:.4f} b={track.parabola[1]:.4f} c={track.parabola[2]:.4f} невязка {track.parabola[3][0]:.5f}', body_style))
        flowables.append(Paragraph(f'Скорость Vy {track.parameters[1]:.4f} м/c', body_style))
        flowables.append(Paragraph(f'Скорость Vx {track.parameters[2]:.4f} м/c', body_style))
        flowables.append(Paragraph(f'Скорость V0 {track.parameters[3]:.4f} м/c', body_style))
        flowables.append(Paragraph(f'Угол взлета {track.parameters[4]:.2f} градусов', body_style))
        flowables.append(Paragraph(f'Радиус частицы {track.parameters[5]:4E} м', body_style))
        flowables.append(Paragraph(f'Масса частицы {track.parameters[6]:.4E} кг', body_style))
        flowables.append(Paragraph(f'Заряд частицы {track.parameters[7]:.4E} Кл', body_style))

        flowables.append(PageBreak())

        # angle_value = track.parameters[4]  # 获取角度值
        # angle_values.append(angle_value)  # 将角度值添加到列表中
        

        #c.setFont("Times", 14)

        #c.drawCentredString(10.5*cm, 28.0*cm, f'Трек №{i}')

        #c.drawImage(image1, 10, 10, width=None,height=None)
        #c.drawImage(image2, 100, 100, width=None,height=None)
        #c.showPage()
    doc.build(flowables)
    
    os.system(f'start report.pdf')
    return angle_values
    

def _draw_q_hist(tracks):
    q = [track.parameters[7] for track in tracks]

    num_bins = 10

    n, bins, patches = plt.hist(q, num_bins,facecolor='blue', alpha=0.5)
    plt.ylabel('Кол-во, ед.')
    plt.xlabel('Q, Кл')
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png', bbox_inches='tight')
    plt.close()
    image_stream.seek(0)
        
    return image_stream

def _draw_tracks_3d(tracks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mins = []
    maxs = []

    for track in tracks:
        points = track.flatten_3d_points
    
        ax.scatter(points[:,0,0], points[:,0,1], points[:,0,2], marker='o')
        mins.append(min(points[:,0,2]))
        maxs.append(max(points[:,0,2]))
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    ax.set_zlim(min(mins) - 2, max(maxs) + 2)

    ax.set_xlabel('X, mm')
    ax.set_ylabel('Y, mm')
    ax.set_zlabel('Z, mm')

    ax.view_init(-70, -90)
        
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png', bbox_inches='tight')
    plt.close()
    image_stream.seek(0)
        
    return image_stream



def _draw_image(image, width=None, height=None):
    image_im_data = io.BytesIO()
    image.save(image_im_data, format='png')
    image_im_data.seek(0)
    #image_out = ImageReader()
    return Image(image_im_data, width, height)



