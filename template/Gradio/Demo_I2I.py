import gradio as gr

def main():

    def demo_fn(input_img):
        #####
        # demo functionk
        # currently just mirror input_img
        return [input_img['image'], input_img['image'], input_img['image'], input_img['mask']]

    def save_fn():
        pass

    ####
    # Gradio
    ####
    with gr.Blocks() as demo:
        gr.Markdown("Img2Img Template")

        # Sliders
        gr.Slider(minimum=0, maximum=10, step=1, value=5, label='Slide1', interactive=True)
        gr.Slider(minimum=0, maximum=10, step=1, value=5, label='Slide2', interactive=True)
        gr.Slider(minimum=0, maximum=10, step=1, value=5, label='Slide3', interactive=True)
        
        # Images
        input_img = gr.Image(label="Input", type='pil', interactive=True, tool='sketch').style(height=256)
        # output_img = gr.Image(label="Output").style(height=256)
        output_img = gr.Gallery(label="Output").style(grid=4, height=256)

        # Examples
        gr.Examples(
            examples=glob(os.path.join('test_imgs', '*.png')),
            inputs=input_img
        )

        # Buttons
        run_btn = gr.Button('Run')
        run_btn.click(fn=demo_fn, inputs=[input_img], outputs=output_img)

        save_btn = gr.Button('Save')
        save_btn.click(fn=save_fn)

    demo.launch(share=True)

if __name__ == "__main__":
    main()
