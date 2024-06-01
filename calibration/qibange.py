from PIL import Image, ImageDraw

def generate_chessboard(width, height, square_size):
    # 创建一个新的图像对象
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # 绘制棋盘格
    for i in range(0, width, square_size):
        for j in range(0, height, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                draw.rectangle([(i, j), (i+square_size, j+square_size)], fill="black")

    return image

# 设置生成图像的参数
width = 700
height = 900
square_size = 100

# 生成棋盘格图像
chessboard_image = generate_chessboard(width, height, square_size)

# 保存图像到文件
chessboard_image.save("chessboard.png")