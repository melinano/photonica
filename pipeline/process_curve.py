from PIL import Image, ImageDraw
from math import floor
import os
from glob import glob
import numpy as np


def rosenblatt_parzen_smoothing(coords, cs):
    numberOfPoints = len(coords)
    ma_cords = []

    for i in range(numberOfPoints):
        sum = 0.0
        sum1 = 0.0
        for j in range(numberOfPoints):
            if i != j:
                z = abs((i - j) / cs)
                #r_b = abs(coords[i][1] - coords[j][1])
                if z <= 1:# and r_b <= 5:
                    sum += coords[j][1] * (1 - z)
                    sum1 += (1 - z)

        ma_cords.append((i, sum / (sum1 + 1e-9)))  # Для избегания деления на ноль добавим небольшую поправку

    return ma_cords


def moving_average(coords, window_size):
    """
    Вычисление скользящего среднего для данных с окном заданного размера

    :param coords: исходный массив координат
    :param window_size: размер окна
    :return: массив с сглаженными координатами
    """
    ma_coords = []

    for i in range(len(coords)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(coords), i + window_size // 2 + 1)
        window = coords[window_start:window_end]
        ma_coords.append((i, sum([y for _, y in window]) // len(window)))
    return ma_coords


def find_sequences(arr, threshold=2):
    sequences = []
    current_sequence = [arr[0]]

    for i in range(1, len(arr)):
        if abs(arr[i] - arr[i-1]) <= threshold:
            current_sequence.append(arr[i])
        else:
            sequences.append(current_sequence)
            current_sequence = []
    sequences.append(current_sequence)

    return sequences

def count_common_elements(arr1, arr2):
    count = 0
    for element in arr1:
        if element in arr2:
            count += 1
    return count


def get_line_coords(image):
    """
    Поиск усредненых y-координат для каждой x-координаты исходного изображения

    :param image: массив данных, описывающий открытое изображение
    :return: координаты кривой, которая представляет собой среднее значение y для каждой x-координаты
    """
    width, height = image.size

    y_dict = {x: [] for x in range(width)}

    sequences = []

    for x in range(width):
        for y in range(height):
            pixel_value = image.getpixel((x, y))
            if pixel_value == (255, 255, 255) or pixel_value == 255:
                y_dict[x].append(y)
        if len(y_dict[x]) != 0:
            sequences = find_sequences(y_dict[x])
        if len(sequences) > 1 and x != 0:
            common_elements = []
            for i in range(len(sequences)):
                    common_elements.append(count_common_elements(sequences[i], y_dict[x-1]))

            y_dict[x] = sequences[common_elements.index(max(common_elements))]
    line_coords = []
    count = 0
    for key, value in y_dict.items():
        if len(value) == 0:
            count+=1
        else:
            for key_, value_ in y_dict.items():
                if key_ < count:
                    y_dict[key_] = value
                else:
                    break
            break
    for key, value in y_dict.items():
        if len(value) == 0 and key != 0:
            line_coords.append((key, line_coords[key-1][1]))
        else:
            line_coords.append((key, (max(value) + min(value)) / 2))
    return line_coords


def create_thin_line_image(image_path, folder_path, is_ma=True):
    """
    Отрисовка изображения кривой (тонкая белая линяя)

    :param image_path: путь к оригинальному изображению
    :param folder_path: путь к результирующему изображению
    :param is_ma: какие координаты использовать
    :return:
    """
    image = Image.open(image_path)

    coords = get_line_coords(image)


    new_image = Image.new("RGB", image.size, color="black")

    for x, y in coords:
        new_image.putpixel((x, floor(y)), (255, 255, 255))

    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{image_filename}_thin.png" if not is_ma else f"{image_filename}_thin_ma.png"
    new_image.save(os.path.join(folder_path, output_filename))


def save_coordinates_to_file(image_path, result_path, coordinates):
    """
    Сохраняет координаты в файл
    :param image_path: путь исходного изображения
    :param result_path: путь к результирующему изображению
    :param coordinates: координаты
    :return:
    """
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{image_filename}_smoothed_coords.txt"

    with open(os.path.join(result_path, output_filename), "w") as file:
        for x, y in coordinates:
            file.write(f"{x}\t{y}\n")

    print(f"{image_filename} was completed")


def process_image(image_path, detect_mode='min', cs=50, prominence_=8):
    """
    Получает координаты 1 изображения и сохраняет в файл

    :param image_path: путь исходного изображения
    :param result_path: путь к результирующему изображению
    :param window_size: размер окна для moving average
    :return:
    """
    image = Image.open(image_path)
    width, height = image.size
    coords_list = get_line_coords(image)
    ma_coords = rosenblatt_parzen_smoothing(coords_list, cs)
    #coords_to_save = []
    #for x, y in ma_coords:
    #    coords_to_save.append((x, height-y))
    #save_coordinates_to_file(image_path, result_path, coords_to_save)
    smoothed_image = Image.new("RGB", image.size, color="black")
    for x, y in ma_coords:
        smoothed_image.putpixel((x, floor(y)), (255, 255, 255))
    # image_filename = os.path.splitext(os.path.basename(image_path))[0]
    # output_filename = f"{image_filename}_postprocessing_smoothing.png"
    # new_image.save(os.path.join(result_path, output_filename))

    from scipy.signal import find_peaks
    # Ваш список координат кривой (массив кортежей)
    y_values = [coord[1] for coord in ma_coords]
    # Найти локальные максимумы
    maxima, _ = find_peaks(y_values, prominence=prominence_)

    # Найти локальные минимумы
    minima, _ = find_peaks([-y for y in y_values], prominence=prominence_)

    first_min = min(minima)
    first_max = min(maxima)

    if first_min > first_max:
        to_add = first_max - abs(first_min-first_max)
        if to_add < 0:
            to_add = 1
        minima = np.insert(minima, 0, to_add)

    last_max = max(maxima)
    last_min = max(minima)

    if last_max > last_min:
        to_add = last_max + abs(last_min - last_max)
        if to_add > len(ma_coords):
            to_add = len(ma_coords)-1
        minima = np.append(minima, to_add)

    distances = []
    sum_dist = 0
    for i in range(len(minima) - 1):
        sum_dist += minima[i+1] - minima[i]
        distances.append(minima[i+1] - minima[i])

    if detect_mode == 'min':
        rect_width = min(distances)
    elif detect_mode == 'average':
        rect_width = sum_dist/(len(minima) - 1)



    # Получить соответствующие координаты (x, y) для максимумов и минимумов
    maxima_coordinates = [ma_coords[i] for i in maxima]
    minima_coordinates = [ma_coords[i] for i in minima]


    new_image = Image.new("RGB", image.size, color="black")
    draw = ImageDraw.Draw(new_image)

    for x, y in ma_coords:
        new_image.putpixel((x, floor(y)), (255, 255, 255))


    rectangles = []

    for x, y in maxima_coordinates:
        point_size = 10
        point_color = (255, 0, 0)

        left_upper = (x - point_size, y - point_size)
        right_lower = (x + point_size, y + point_size)

        draw.ellipse([left_upper, right_lower], fill=point_color, width=2)
        rect_start = (x - rect_width//2, y - 100)
        rect_end = (x + rect_width//2, y + 50)

        rectangle_color = (0, 0, 255)  # RGB для синего
        draw.rectangle([rect_start, rect_end], outline=rectangle_color, width=2)
        cur_rect = []
        for x_,y_ in ma_coords:
            if x_ >= x - rect_width//2 and x_ <= x + rect_width//2:
                cur_rect.append(y_)
        if abs(len(cur_rect) - rect_width) < 3:
            rectangles.append(cur_rect)
    result_rect = []
    for idx in range(len(rectangles[0])):
        result_rect.append(int(sum([rect[idx] for rect in rectangles]) / len(rectangles)))

    # for idx in range(len(result_rect)):
    #     new_image.putpixel((width//2+idx-len(result_rect)//2, result_rect[idx]+height//3), (255, 255, 255))

    for x, y in minima_coordinates:
        point_size = 10
        point_color = (0, 255, 0)

        left_upper = (x - point_size, y - point_size)
        right_lower = (x + point_size, y + point_size)

        draw.ellipse([left_upper, right_lower], fill=point_color, width=2)

    return smoothed_image, new_image, result_rect

    # image_filename = os.path.splitext(os.path.basename(image_path))[0]
    # output_filename = f"{image_filename}_postprocessing.png"
    # new_image.save(os.path.join(result_path, output_filename))


def process_images_in_folder(folder_path, result_path, window_size=3):
    """
    Получает координаты всех изображений из папки и сохраняет в файлы

    :param folder_path: путь к папке содержащей исходные изображения
    :param result_path: путь к результирующему изображению
    :param window_size: размер окна для moving average
    :return:
    """
    images_path = glob(os.path.join(folder_path, "*.png"))

    for image_path in images_path:
        process_image(image_path, result_path, window_size)


if __name__ == '__main__':
    # "/путь/к/папке/с/изображениями/"
    folder_path = "/Users/danielageev/Work/AI BMSTU/ФОТОНИКА/PHOTOS/NEW_OCTOBER/ORIGINAL/part2/1stage/all_masked"
    # "/путь/к/одному/изображению/"
    image_path = "/Users/danielageev/Work/AI BMSTU/ФОТОНИКА/pipeline/predictions/S4_SEM#3 cropped_assembled.png"
    # "/путь/к/папке/с/результатами/"
    result_path = "/Users/danielageev/Work/AI BMSTU/ФОТОНИКА/PHOTOS/coords"
    # Запись координат в txt
    #image = Image.open('/Users/danielageev/Work/AI BMSTU/ФОТОНИКА/pipeline/predictions/S4_SEM#2_7 cropped_assembled.png')
    #print(get_line_coords(image))
    process_image(image_path, result_path, 20)
    process_image("/Users/danielageev/Work/AI BMSTU/ФОТОНИКА/pipeline/predictions/S4_SEM#2_7 cropped_assembled.png", result_path, 20)# Получаем 1 txt файл из 1 изображения
    #process_images_in_folder(folder_path, result_path, 0.5)  # Получаем txt файлы из всех изображений в папке
    
    # Отрисовка
    # Результирующее изображение отрисовывается по сглаженным координатам
    #create_thin_line_image(image_path, result_path, True)
    # Результирующее изображение отрисовывается по усредненным координатам
    #create_thin_line_image(image_path, result_path, False)

