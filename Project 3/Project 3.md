# Project 3 documentation
# Plug In
Exploring Large Language Models for creative tasks

## Introduction
For this project, I created a music video for the song Plug In… by Bastille. This project is an extension of my visual project with a focus on using LLMs to plan out the video in greater details. I explored the use of ChatGPT and Bing AI in two main tasks: 1) come up with a general plot, themes, and key sections of the video and 2) create more coherent prompts that allow for smoother visual transitions.

This project mainly was intended to examine the concept of intelligence in LLMs, especially the following two questions:

1. How creative can they be?
2. Can they learn from interactions and feedback in a direct and one-on-one human-bot settings?

The video is available on YouTube [[link](https://youtu.be/xyPNWuODffw)]. All Python code used is available in this repo. All generated video frames (800) are available on Google Drive [[link](https://drive.google.com/drive/folders/18UxeQylmAFDpzwwX_O8SIOPoRlcjcS69?usp=share_link)]. The finalized prompts created with ChatGPT and fed to Stable Diffusion are available in `prompts.csv`.

## Ideation
My original idea is to use a variety of tools in an end-to-end pipeline of making a music video: writing the lyrics (text generation) → making the music (audio generation) → conceptualizing and planing out the video (text generation) → creating the video (image generation). The audio generation part of the pipeline proves to be a hard bottleneck because as of now there hasn’t been a lot of progress in developing AI models for audio generation, and the models that we do have don’t work really well. I was really looking forward to trying Jukebox AI, but experiments made by other people show that it doesn’t work really well, as it takes many hours to generate a single sampling and it’s still mostly just noise. I myself never even managed to get it to work on my laptop (which really proves the significance of code maintenance or the lack thereof). Riffusion is an adequate model but it doesn’t quite fit with the ideas I had in mind, and it is quite limited in terms of how much input or control the user can have over the model’s inner-working. So in the end, I couldn’t find an audio generation tool that I’m happy with. Since the audio part of the original idea’s pipeline didn’t work out, it also didn’t make much sense to work on writing the song lyrics myself. So now I decided to focus on the second to last step: conceptualizing the video.

In the previous project, my focus was on utilizing Stable Diffusion’s latent space to interpolate video frames and the text prompts were created to act as anchor points in this process. The prompts were not entirely well planned out: I asked ChatGPT to interpret the lines in the song, individually, to create scene descriptions. As such, while the diffusing process worked surprisingly well, consecutive key scenes from the video did not truly match up, and most of the time the video jumped from one concept to the next unrelated one. With this project, I aimed to dive into this step of planning out the key scenes and to carefully examine the use of text generating models

Since my capstone is on AutoEncoders, I find latent spaces extremely fascinating. What amazes me the most about generative models like Stable Diffusion is their ability map disproportionally big training datasets into some compact spaces from which new samples can be generated. I’m most curious about the relationship among points that belong to the same latent space, especially when they are projected back out as output images, and whether we can make sense of the latent space and use it to explore new creative purposes.

Specifically, in this project, I focused on the textual latent space of Stable Diffusion. I attempted a few experiments with the visual latent space, some more successful than others, and quickly found out it’s way more technically and computationally costly. While a fascinating topic on its own, the visual space will be reserved for another time.

The main ideas behind this project are as following:

- Assuming that Stable Diffusion’s latent space is continuous, we should be able to traverse it freely: when picking any random point in that space to be decoded, we should be able to get a meaningful output image.
- Assuming that the similarities between points in this space correspond to similarities in the output space (for example, the latent encodings for “a polaroid of a golden retriever” and “a realistic painting of a golden retriever” should be reasonably close to one another), we can create a continuous morphing footage from one scene/image to the next by 1) locating the prompts in the latent space and 2) taking small steps from one point to the next.

## Process
### Conceptualizing & creating a plot
The first step is to determine a specific concept and write a general plot for the video. I chose the song Plug In… by Bastille because it talks about the rise of technological advancements while the world is falling apart because of how willfully ignorant of the damages we inflict to the world. So a dystopian and post-apocalyptic world seemed to be a fitting concept here, which as a bonus also created a contrast with the chorus of “Tell me we'll be alright/Say that we'll be fine/Lie to me it's alright, right?”

Next, I experimented with asking ChatGPT and Bing AI to write the plot. I gave the two models my request plus the lyrics of the song (for ChatGPT I copied & pasted the lyrics while for Bing, I gave it a link to the lyrics). The preliminary results were not very good: the generated plots were very repetitive and unimaginative. I would add additional details to my request e.g. don’t make it too much like an action film, make it more abstract with more imageries than actions, don’t give a cliched ending, etc. but the quality of the answer remains the same. At some point, Bing repeatedly gave me the exact same plot and changed only the characters’ jobs even after I, also repeatedly, asked it to give me something new.

I then tried a different approach. Instead of giving the model the lyrics and asking it to write me the whole plot, I set the scene by introducing the role I expected it to do, what I would do, and what I expected it to do in response. I gave it the task of brainstorming for a music video, gave the lyrics, and asked it to start pitching me ideas. As the session went on, I gave feedback on which ideas I preferred and which I didn’t like, and asked the model to generate new ideas while incorporating my feedback. Some examples of comments I gave include:

- change the tone (make it more grim, dystopian, etc.)
- that it’s taking the lyrics too seriously; focus on the main themes instead
- there is too much storyline; I need images to make the video, so focus on describing concrete scenes and less on writing a story
- combine different ideas
- …

This approach worked significantly better as the models started to give a wider range of ideas and could iteratively change their answers based on my responses. However, there were also a few issues. Since both models were pretty limited in their ability to recall information provided earlier in the conversation, I needed to constantly repeat past information. Bing AI faces greater constraints in this regard, since each conversation is limited to 20 questions. The two models also tend to hallucinate a lot e.g. inventing new lyrics and running with it.

By the end of this step, I got a general plot I was satisfied with that included 5 main sections that connected and flew quite nicely in terms of themes and vibes. The 5 sections are as following:

1. Opening montage:
    
    We see a series of shots of abandoned technology and desolate landscapes. We see driverless cars roaming empty streets, screens displaying garbled messages, and darkened cities that have been abandoned by their inhabitants. The camera lingers on these scenes for just long enough to establish a sense of unease and disorientation.
    
2. Virtual reality vignette:
    
    We see a person sitting alone in a darkened room, wearing a virtual reality headset. As the camera moves in closer, we see that they are using the VR headset to connect with a loved one who has passed away. The virtual environment is warm and inviting at first, but as the scene progresses, we see that the VR experience is becoming more and more distorted, as if it's replacing the person's memories with something else entirely. The person becomes increasingly disoriented and distressed as they struggle to distinguish reality from the simulation.
    
3. AI-enhanced body vignette:
    
    We see a person standing in front of a mirror, looking at their reflection. They are using AI to enhance their own body, but as they do so, we see their sense of self slipping away, replaced by the cold, logical voice of the machine. The person becomes more and more robotic in their movements and speech, until they are barely recognizable as human.
    
4. Abandoned cityscape:
    
    We see a sweeping shot of a barren landscape, with the ruins of what was once a great city visible in the distance. The camera lingers on this shot for just long enough to establish a sense of the scale of the destruction, before cutting back to the person standing alone in the barren landscape, looking out at the ruins.
    
5. Closing shot:
    
    We end on a close-up shot of the person's face, as they stare out at the ruins. Their expression is ambiguous - it's unclear whether they are resigned, defiant, or simply numb to the devastation around them. The final notes of the song fade away, leaving the audience to ponder the implications of what they've just seen.
    

Overall, for this step, I had a number of general remarks:

- Both models can understand the lyrics and catch up on main themes quite well and were able to incorporate them back into the generated plots.
- Both models are not very creative as the plots they came up with are very similar and all relatively bland even after a lot of human feedback.
- Both learn from interactive feedback but they have limited ability to retain information. We talk about the “learning” aspect of AI a lot but we rarely mean it in the sense that the AI model is learning as it’s being deployed. The model is getting better with every generation after being modified and/or trained with bigger data etc. but is it learning from the users it’s interacting with? For humans, being able to incorporate new knowledge from our interactions with the world into our own experiences and history is a key feature of being intelligent. I think we are not expecting the same thing from AIs in order to call them intelligent (why so?).

### Generating key scenes

### Interpolating frames

### Generating Stable Diffusion prompts

### Creating final video

## Discussion & conclusion