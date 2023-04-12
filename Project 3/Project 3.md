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
Having the general plot with 5 main sections, I moved on to creating descriptions of key scenes for each section. For this step and all steps following this, I used ChatGPT. So for each section, I asked it to give me descriptions of a few key scenes within that section, with the quantity of scenes being up to it to decide. The longer the conversation went, the more it was inclined to forget earlier information and go completely off topic, so I kept reminding it. A query would look like this:

> Let's dive deep into each section of our plot and plan it out in great details. I will give you the general plot of each section, and you give me descriptions of key scenes for that section. The section I want you to work on is [description of section].

By the end of this step, the sections were pretty well planned out with a good number of key scenes.

### Interpolating frames
Recall that in the previous project, I conducted frame interpolation by converting a few prompts into numerical arrays and taking small steps from one array to the next. While this works fine with Stable Diffusion, it’s purely math so the result was visually very abstract. In this project, I wanted to generate more prompts in between to create an even smoother transition. Think of it as similar to array interpolation but instead of only math, I wanted the interpolation to make sense logically. I’m not sure if there is a term for it, so I call this semantic interpolation. The idea is that if I have two prompts describing scene A and scene B, can I ask ChatGPT to give me descriptions of what can possibly happen between those two scenes that make logical sense?

This was an interesting experiment because there is no easy way to explain exactly what I wanted it to do. So I turned to examples. I gave examples of two prompts I wanted to conduct frame interpolation plus the interpolated frames in between then asked ChatGPT to do the same thing:

> I want you to do something that I call frame interpolation. Here is how it works. I will give you descriptions of two consecutive key scenes and your job is to provide me with descriptions of frames that will fit in between those two prompts. Here is an example of frame interpolation (from 0 to 6) I want you to do:
> 
> 0: "a dystopian warzone with zombies, dark clouds in the sky, night",
> 
> 1: "a dystopian warzone, dark clouds in the sky",
> 
> 2: "a dystopian city, clouds in the sky",
> 
> 3: "a city",
> 
> 4: "a beautiful city",
> 
> 5: "a beautiful and modern city with blue sky",
> 
> 6: "a beautiful and modern city with blue sky and green trees"
> 
> Now I want to interpolate between the following two scenes:
> "A deserted cityscape, with tall skyscrapers casting long shadows over the empty streets below"
> 
> and
> 
> "A driverless car moves slowly through an abandoned street, its headlights piercing through the thick fog that envelops the scene"
> 
> Give me a 4-step frame interpolation like the example. Keep the descriptions short and succinct like the examples I gave.

After some further feedback, it could generally understand the task. For example:

Prompts:

FROM

*"A driverless car moves slowly through an abandoned street, its headlights piercing through the thick fog that envelops the scene”*

TO

*"A close-up shot of a robotic arm repeatedly pressing a button on an abandoned control panel. The camera then pans out to show a large, empty factory floor with rows of deactivated machines."*

Interpolated frames:

> 0: "A driverless car moves slowly through an abandoned street, its headlights piercing through the thick fog that envelops the scene"
> 
> 1: "The car drives deeper into the fog, its headlights illuminating the abandoned buildings on either side"
> 
> 2: "The car passes by a graffiti-covered wall with faded colors, barely visible in the fog"
> 
> 3: "The car emerges from the fog and drives past rows of deactivated machines on an abandoned factory floor"
> 
> 4: "A close-up shot of a robotic arm repeatedly pressing a button on an abandoned control panel"
> 
> 5: "The camera pans out to show the deserted factory floor with rows of deactivated machines"
> 
> 6: "The camera continues to pan out, revealing the abandoned industrial complex"
> 
> 7: "The camera pulls back further, showing the desolate and overgrown wasteland that surrounds the complex."

Two main remarks emerged from this step:

- ChatGPT learns surprisingly well when provided with specific examples. I’d assume that to ChatGPT, the task itself is new and unseen before (I am having a hard time explaining what I want to do here). This is a similar process to teaching a human to learn: first by giving instructions, giving examples to clarify expectations, and giving feedback for improvements. ChatGPT is showing, in my opinion, pretty remarkable ability to learn new tasks from this process just as a human would.
- The memory issue is still there, so I needed to give it examples every time.

### Generating Stable Diffusion prompts
As the model is not the only entity learning from feedback in this whole process, I too have learned that giving examples worked better than giving pure instructions without good clarifications. I gave it specific examples of good Stable Diffusion prompts and asked it to generate Stable Diffusion prompts based on frame description it gave before:

> I will give you descriptions and I want you to turn them into Stable Diffusion prompts. A prompt includes the original description and additional descriptors. An example is this: The original description is "The camera cuts to a darkened city street where a lone figure in a hooded coat walks past flickering streetlights" and the corresponding Stable Diffusion prompt is "Dark and gritty cityscape, ominous atmosphere, hooded figure walking in the night, low-key lighting, film noir-inspired, 1080p resolution, moody". 

By the end of this step, I had a total of 81 prompts (the video is 2:40s long = 160s, which means I had 1 prompt for every 2 seconds, compared to the previous project: 48 prompts / 5:53s = 353s ~ 1 prompt every 7 second).

### Creating final video
The rest of the process (encoding text prompts, interpolating mathematically, and generating images) was conducted using the same pipeline as [my previous project](https://github.com/nhungoc1508/S23-IM3312-ArtIntel/tree/main/Project%202).

## Discussion & conclusion
