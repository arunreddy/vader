


def  plot_results():
      _, te_Y = dataset.fetch_test_instance(test_idx)
    figure = plt.figure()
    print('===>',te_Y)
    if te_Y == 1:
        label = 'domestic'
    else:
        label = 'wild'
    img_path =  os.path.join(images_path,dataset.test_file_names[test_idx])
    img=mpimg.imread(img_path)
    plt.title(label)
    plt.imshow(img)

    columns=3
    rows=3
    figure = plt.figure(figsize=(columns * 3, rows * 3))
    for i in range(columns * rows):
        figure.add_subplot(rows, columns, i + 1)
        img=mpimg.imread(data[i][0])

        print(data[i][2])

        if data[i][2] == 1:
            label = 'domestic'
        else:
            label = 'wild'

        
        label += ' [{:0.3f}]'.format(data[i][1])
        
        plt.title(label)
        plt.imshow(img)
        

    plt.show()