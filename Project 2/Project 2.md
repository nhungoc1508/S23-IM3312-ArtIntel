# Project 1 documentation
# Diffusing Bohemian Rhapsody
Video interpolation and an exploration of the Stable Diffusion latent space

<p align="center"><img src="https://github.com/nhungoc1508/S23-IM3312-ArtIntel/blob/main/Project%202/showcase_rand_5.gif" width=500/></p>

## Introduction
For this project, I created a music video for the song Bohemian Rhapsody by Queen. Using about 50 prompts as keyframes, I experimented with exploring Stable Diffusion’s textual latent space to interpolate between said frames. The final result is a continuously morphing video depicting a re-imagination of the song lyrics by ChatGPT and Stable Diffusion. By taking a closer and low-level look into the architecture of Stable Diffusion, we can start to unravel the black box of generative algorithms and to experiment with new creative ends.

The final video has been uploaded to YouTube [[link](https://youtu.be/RxNYet1o3eM)]. The video itself contains hardcoded song lyrics and the corresponding Stable Diffusion prompts. All the Python code used is in this repo. If possible, it is more optimal to run the notebooks on GPU. All the generated video frames (1170 frames in total) are available on Google Drive [[link](https://drive.google.com/drive/folders/1ivISTXaGFLW2trX9MaWk4wp-_a4DTs8B?usp=share_link)].

## Ideation
Since my capstone is on AutoEncoders, I find latent spaces extremely fascinating. What amazes me the most about generative models like Stable Diffusion is their ability map disproportionally big training datasets into some compact spaces from which new samples can be generated. I’m most curious about the relationship among points that belong to the same latent space, especially when they are projected back out as output images, and whether we can make sense of the latent space and use it to explore new creative purposes.

Specifically, in this project, I focused on the textual latent space of Stable Diffusion. I attempted a few experiments with the visual latent space, some more successful than others, and quickly found out it’s way more technically and computationally costly. While a fascinating topic on its own, the visual space will be reserved for another time.

The main ideas behind this project are as following:

- Assuming that Stable Diffusion’s latent space is continuous, we should be able to traverse it freely: when picking any random point in that space to be decoded, we should be able to get a meaningful output image.
- Assuming that the similarities between points in this space correspond to similarities in the output space (for example, the latent encodings for “a polaroid of a golden retriever” and “a realistic painting of a golden retriever” should be reasonably close to one another), we can create a continuous morphing footage from one scene/image to the next by 1) locating the prompts in the latent space and 2) taking small steps from one point to the next.

## Process
### Generating descriptions
I used a number of prompts as keyframes for the video. Essentially, these prompts serve as the main scenes of the video, and the exploration of Stable Diffusion’s latent space aimed to fill in the gaps among them.

For this task, I asked ChatGPT to provide its interpretations of Bohemian Rhapsody’s lyrics and generate succinct descriptions of scenes and images for each line. I tested out a few different queries, each one with a bit more details on what I wanted based on ChatGPT’s previous answers. For example, I asked it to be concise, avoid describing actions, and be as descriptive as possible. The query is as following:

> *I am reimagining a music video for the song Bohemian Rhapsody by the band Queen. Plan out the different scenes and images for it by, for each of the following lines from the song, provide a description of a scene or image to associate with it. For each line, give a short, succinct description of what the scene will be. For example, for the line "Any way the wind blows doesn't really matter to me, to me" the associated scene description can be something like "a person standing on a cliff facing the forest with trees bending to strong winds". In other words, associate each line with a specific scene or image and describe that in detailed and succinct language. The scenes should be described as still scenes, rather than with motions. Focus on describing the scenes and images, rather than saying what the images imply. The lines are as follows:
[Song lyrics]*

ChatGPT then gave me a list of 48 scene descriptions associated with the lines of the song lyrics. These descriptions were then used as the prompts to Stable Diffusion.

### Randomizing styles
Before feeding all scene descriptions to Stable Diffusion, I added some art styles to each of the prompts. I collected the style keywords from the Easy Diffusion server from all of the categories except for emotions (I don’t really see their effects). The final list of styles consisted of 190 different keywords (e.g. photoshoot, constructivist, visual novel, electric colors, etc. as well as a lot of different artists). I assigned each description 4 random keywords from the list to create the final prompt, for example: “A shot of a person looking into a mirror, unsure of what is real and what is not, photoshoot, pyrography, constructivist, Ko Young Hoon.” This process was conducted with the random function of Python. As such, there are many prompts with style combinations that don’t really make sense or are self-contradicting, like “photoshoot, watercolor” or many different artists at once. I didn’t make further adjustments to keep it as randomized as possible.

The style keywords are stored in `styles.json`. The script for randomizing styles is in `Prompts randomization.ipynb`. The final list of prompts are available in `prompts.csv`

### Interpolating latent encodings

Since I needed to get a bit closer to the underlying pipeline of Stable Diffusion and to generate a lot of images, the available UI was not the best option. I found two Python libraries with some form of Stable Diffusion implementations that I could use: [KerasCV](https://keras.io/keras_cv/) and [Diffusers](https://huggingface.co/docs/diffusers/v0.14.0/en/index). Both were very useful though both took a lot of debugging to work properly. Even though Diffusers was excellent for exploring the visual latent space, I opted to use KerasCV because I found [this tutorial](https://keras.io/examples/generative/random_walks_with_stable_diffusion/) that was precisely on what I was trying to do and so it took me less time to make it work properly.

All remaining code, especially related to this step, can be found in `Stable Diffusion interpolation.ipynb`.

The process of getting interpolated encodings for the video is as following:

#### 1. Encoding keyframes:

I fed the prompts finalized in the previous step to the encode_text() function of the Stable Diffusion model. For each prompt, the model returned an encoding of the prompt in the form of a 77 x 768 matrix. This is the projection of that keyframe onto the textual latent space. I obtained the encodings for all keyframes this way.

#### 2. Interpolating between keyframes:

Since each prompt was now represented by a numerical matrix, I could take small and equal steps from one keyframe to the next by interpolating between different matrices.

One thing I needed to decide on here was how many steps to take between two keyframes. I chose the frame rate for my video to be 3 frames per second, since it would give a smooth enough transition between the frames and would amount to a reasonable total amount of frames needed (the song is about 6 minutes long, so 3fps would need about 6 * 60 * 3 = 1080 frames). Then I identified the timestamps for the keyframes i.e. where the lyrics start and end. This would decide on how much time there is between one keyframe to the next, and thus how many steps to make. For example, these two consecutive lines from the song:

> “Caught in a landslide, no escape from reality” (0:08 - 0:16) = frame_0

> “Open your eyes, look up to the skies and see” (0:16 - 0:26) = frame_1

The timestamps of frame_0 and frame_1 were the starts of the lines (0:08 and 0:16), so the distance between them would be 8 seconds. Since frame rate was 3fps, I would take 8 * 3 = 24 steps from the encoding of frame_0 to that of frame_1. Using the interpolation function, I would be able to obtain 24 additional matrices representing a relatively smooth transformation from frame_0 to frame_1. Theoretically, these in-between encodings should project back out into images that are quite similar to one another, becoming less and less similar to the starting keyframe and more and more similar to the destination keyframe — filling in the gaps.

I followed this process for every pair of consecutive keyframes. When all stacked together, the collective encodings for the entire video from start to finish was one big matrix of size 1170 x 77 x 768, corresponding to 1170 encodings of size 77 x 768 each. (1170 > 1080 frames as per my calculations above, and indeed there were some mistakes in my code. I realized this while editing the final video.) Overall, this was a good number of encodings. It was then time to generate the images.

### Generating images
The interpolated encodings were then fed to the model. KerasCV was very helpful in providing the different functions I could use to my purposes rather than simply feeding the text prompts. Essentially, I was able to hijack the prompt encoding process to make manual adjustments to the latent encodings themselves without the need for the actual text prompts: I started with 48 text prompts and created over 1000 latent encodings. Now I could feed these encodings straight to the decoder to get the final images.

I encountered some hardware incompatibilities on my laptop so I ran the model on Google Colab. The decoding process exhausted GPU RAM very quickly, so I had to divide the encodings into small batches and manually iterate through them. After many sessions, I was able to obtain the final 1170 images created from keyframe prompts and interpolated latent encodings. Each image was of size 640 x 512 pixels. I wanted to make them bigger and with a higher width:height ratio, but other dimensions and sizes depleted the RAM very quickly. Also due to computational constraints, I used 15 inference steps for each image, so the images might look like they’re still in the middle of being adjusted and perfected.

### Creating final video

The last step was to stitch all the images together to produce the video. I used Davinci Resolve for this, and made some editing with changing the speed at some parts to make the whole thing coherent. As mentioned above, there were some mistakes in my calculations which led to the wrong interpolation steps. There were some parts in the video where the frames obviously do not match the displayed lyrics, and this was due to such miscalculations. Since I only detected those after I had created all the images (1170 of them D:), I decided not to re-run the script and tried to fix it a bit in post-editing.

## Discussion & conclusion
It is worth noting that the textual latent space of Stable Diffusion is very high dimensional (77 x 768, which in my understanding is equal to 77 vectors each of size 768) and complicated, and so is the process of using this space to guide its generation of new images. This method of interpolating between matrices is therefore a theoretically and mathematically very simple process compared to the complexity of the space itself. Nevertheless, the result does show the visual transitions between keyframes that I wanted to see. Such transitions were not only from one scene/image/object to the next, but also between different styles, color palettes, etc. We can see that Stable Diffusion’s latent space is not a completely random space: by taking steps from one point to another, the visual transitions are clear and observable. This can be one step closer to more transparent AI systems that we can try to observe, scrutinize, and make sense of.

This project also explores one way in which an image generation system like Stable Diffusion can be employed to generate creative products of higher dimension like videos.

[TBC]
