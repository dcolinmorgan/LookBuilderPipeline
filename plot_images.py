from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread


def showImagesHorizontally(list_of_files, prompt,negative_prompt,model,height, width, controlnet_conditioning_scale,num_inference_steps,guidance_scale,generator,output_path='output.png'):
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = (list_of_files[i])
        imshow(image,cmap='Greys_r')
        axis('off')
    # Add text to the image
    fig.text(0.5, 0.01, f"Prompt: {prompt} \n Neg_Prompt: {negative_prompt} \n Model: {model} \n height: {height} width: {width} \n cond_scale: {controlnet_conditioning_scale} steps: {num_inference_steps} \n guidance: {guidance_scale} seed: {generator}", ha='center', fontsize=10, color='black')

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure
    plt.close(fig)  # Close the figure to free up memory
