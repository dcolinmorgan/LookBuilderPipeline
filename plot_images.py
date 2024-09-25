from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

def showImagesHorizontally(list_of_files, text,output_path='output.png'):
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = (list_of_files[i])
        imshow(image,cmap='Greys_r')
        axis('off')

    fig.text(0.5, 0.01, f"Prompt: {prompt} | Neg_Prompt: {negative_prompt} | Model: {model} | steps: {num_inference_steps} |
            guidance: {guidance_scale}, seed {generator}", ha='center', fontsize=10, color='black')

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure
    plt.close(fig)  # Close the figure to free up memory
