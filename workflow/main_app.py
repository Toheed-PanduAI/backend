import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import OpenAI
import warnings
from pathlib import Path
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, TextClip, CompositeVideoClip, CompositeAudioClip
import moviepy.audio.fx.all as afx
from moviepy.video.tools.drawing import color_split
import image_effects
import audio_prompts
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, play
import scripts.speech_synthesis as speech_synthesis
import random
from utils.util import text_styles, effect_images
from dotenv import load_dotenv

load_dotenv() 

OPEN_AI_SECRET_KEY = os.getenv('OPEN_AI_SECRET_KEY')
ELEVEN_LABS_SECRET_KEY = os.getenv('ELEVEN_LABS_SECRET_KEY')
STABILITY_SECRET_KEY = os.getenv('STABILITY_SECRET_KEY')

warnings.filterwarnings("ignore", category=DeprecationWarning)  

client = OpenAI(api_key=OPEN_AI_SECRET_KEY)

elevan_labs_client = ElevenLabs(
  api_key= ELEVEN_LABS_SECRET_KEY
)

effects = random.choice(list(effect_images.values()))

filenumber = 43

user_series = {
    "user_prompt": """Video Number	Topic	Title	Duration	Script	Screenplay	Key Visuals	Character Involved	Engagement	Background Music	Special Effects	Animation Style	Seed Text	Call to Action	Language	TG	Tags	Video Description	Detailed Description	Character Descriptions	Screenplay
1	Creation of the World	Creation of the World - Summary	40-60 seconds	Narrator: 'In the beginning, God created the heavens and the earth. Over six days, God created light, sky, land, vegetation, stars, sea creatures, birds, animals, and mankind. On the seventh day, He rested.'	Visual montage of the creation process: cosmos, earth forming, plants growing, animals appearing, and humans created.	God commanding creation, light piercing darkness, animals in their habitats, Adam and Eve.	God - The omnipotent creator. Adam - The first man. Eve - The first woman.	Background Audio: Majestic, awe-inspiring orchestral music.	Majestic orchestral instrumental music	Visual effects to enhance the creation moments	Traditional Animation	In the beginning, God created the heavens and the earth.'	Explore the wonders of God's creation.	English	General Audience	Creation, Genesis, Bible Stories	Experience the creation of the world in a brief and detailed summary.	A summary of God's creation of the world in six days, culminating in a day of rest.	God - The omnipotent creator. Adam - The first man. Eve - The first woman.	Visual montage of the creation process: cosmos, earth forming, plants growing, animals appearing, and humans created.
2	Adam and Eve in Eden	Adam and Eve in Eden - Part 1	40-60 seconds	Narrator: 'God placed Adam and Eve in the Garden of Eden, a paradise filled with everything they needed. They lived in harmony with all creation.'	Visuals of lush garden, Adam and Eve exploring, interacting with animals.	Adam and Eve walking in the garden, interacting with animals, the beauty of Eden.	Adam - The first man, caretaker of Eden. Eve - The first woman, companion to Adam.	Background Audio: Serene, peaceful instrumental music.	Serene instrumental music	Visual effects to enhance the beauty of Eden	Traditional Animation	God placed Adam and Eve in the Garden of Eden.'	Witness the perfect harmony of Eden.	English	General Audience	Eden, Adam and Eve, Bible Stories	Discover the harmony and beauty of Eden where Adam and Eve lived.	God places Adam and Eve in the Garden of Eden, where they live in harmony with all creation.	Adam - The first man, caretaker of Eden. Eve - The first woman, companion to Adam.	Visuals of lush garden, Adam and Eve exploring, interacting with animals.
3	The Fall of Man	The Fall of Man - Part 1	40-60 seconds	Narrator: 'In the Garden of Eden, the serpent tempted Eve to eat the forbidden fruit. She shared it with Adam, and they both disobeyed God.'	Visuals of the serpent, Eve taking the fruit, sharing with Adam.	The serpent, the forbidden fruit, Adam and Eve eating the fruit.	Serpent - The deceiver. Adam - The first man, now fallen. Eve - The first woman, now fallen.	Background Audio: Tense, foreboding instrumental music.	Foreboding instrumental music	Visual effects to enhance the temptation scene	Traditional Animation	The serpent tempted Eve to eat the forbidden fruit.'	Learn about the consequences of disobedience.	English	General Audience	Fall, Temptation, Bible Stories	Understand the first act of disobedience and its consequences.	The serpent tempts Eve to eat the forbidden fruit, leading to disobedience and the fall of man.	Serpent - The deceiver. Adam - The first man, now fallen. Eve - The first woman, now fallen.	Visuals of the serpent, Eve taking the fruit, sharing with Adam.
4	The Fall of Man	The Fall of Man - Part 2	40-60 seconds	Narrator: 'After eating the fruit, Adam and Eve realized their nakedness and felt shame. They hid from God, who confronted them about their disobedience.'	Visuals of Adam and Eve hiding, God confronting them.	Adam and Eve hiding, God's presence, the confrontation.	God - The omniscient judge. Adam - The first man, ashamed. Eve - The first woman, ashamed.	Background Audio: Somber, reflective instrumental music.	Somber instrumental music	Visual effects to enhance the confrontation scene	Traditional Animation	Adam and Eve realized their nakedness and felt shame.'	Understand the impact of the first sin.	English	General Audience	Sin, Shame, Bible Stories	Learn about the immediate effects of Adam and Eve's sin.	Adam and Eve realize their nakedness and feel shame. They hide from God, who confronts them about their disobedience.	God - The omniscient judge. Adam - The first man, ashamed. Eve - The first woman, ashamed.	Visuals of Adam and Eve hiding, God confronting them.
5	Cain and Abel	Cain and Abel - Part 1	40-60 seconds	Narrator: 'Adam and Eve had two sons, Cain and Abel. Cain was a farmer, and Abel was a shepherd. They both offered sacrifices to God.'	Visuals of Cain and Abel, their sacrifices.	Cain with crops, Abel with sheep, their offerings.	Cain - The firstborn son, a farmer. Abel - The second son, a shepherd.	Background Audio: Calm, neutral instrumental music.	Neutral instrumental music	Visual effects to enhance the offerings	Traditional Animation	Cain and Abel offered sacrifices to God.'	Discover the story of Cain and Abel.	English	General Audience	Sacrifice, Cain and Abel, Bible Stories	Explore the differing offerings of Cain and Abel.	Adam and Eve's sons, Cain and Abel, offer sacrifices to God, each reflecting their respective vocations.	Cain - The firstborn son, a farmer. Abel - The second son, a shepherd.	Visuals of Cain and Abel, their sacrifices.
6	Cain and Abel	Cain and Abel - Part 2	40-60 seconds	Narrator: 'God accepted Abel's offering but not Cain's. In jealousy and anger, Cain killed Abel and lied to God about it.'	Visuals of Cain's anger, the murder, God confronting Cain.	Cain's angry face, Abel's death, God's confrontation.	Cain - The first murderer. Abel - The innocent victim.	Background Audio: Dark, intense instrumental music.	Intense instrumental music	Visual effects to enhance the murder scene	Traditional Animation	Cain killed Abel out of jealousy.'	Reflect on the consequences of jealousy and anger.	English	General Audience	Jealousy, Murder, Bible Stories	Reflect on the tragic outcome of Cain's jealousy.	Cain, jealous of Abel's favored offering, kills Abel and lies to God about it, leading to his punishment.	Cain - The first murderer. Abel - The innocent victim.	Visuals of Cain's anger, the murder, God confronting Cain.
7	Noah's Ark	Noah's Ark - Part 1	40-60 seconds	Narrator: 'The world became corrupt, and God decided to cleanse it with a great flood. He instructed Noah, a righteous man, to build an ark.'	Visuals of a corrupt world, God speaking to Noah, the ark construction.	Noah building the ark, corrupt people, God's command.	Noah - A righteous man chosen by God. God - The judge of the world.	Background Audio: Dramatic, anticipatory instrumental music.	Anticipatory instrumental music	Visual effects to enhance the ark construction	Traditional Animation	God decided to cleanse the world with a great flood.'	Witness Noah's obedience to God.	English	General Audience	Noah, Flood, Bible Stories	Witness Noah's faith and obedience in building the ark.	God instructs Noah to build an ark to survive the great flood that will cleanse the world of its corruption.	Noah - A righteous man chosen by God. God - The judge of the world.	Visuals of a corrupt world, God speaking to Noah, the ark construction.
8	Noah's Ark	Noah's Ark - Part 2	40-60 seconds	Narrator: 'Noah built the ark, gathered his family, and brought pairs of animals aboard. The rains began, and the great flood covered the earth.'	Visuals of animals entering the ark, the beginning of the flood.	Animals boarding the ark, Noah's family, heavy rains.	Noah - The faithful builder. Animals - Diverse species saved from the flood.	Background Audio: Heavy, immersive instrumental music.	Immersive instrumental music	Visual effects to enhance the flood scene	Traditional Animation	The rains began, and the great flood covered the earth.'	See the fulfillment of God's warning.	English	General Audience	Ark, Animals, Bible Stories	See the gathering of animals and the start of the flood.	Noah builds the ark, gathers his family and pairs of animals, and the rains begin, covering the earth with a great flood.	Noah - The faithful builder. Animals - Diverse species saved from the flood.	Visuals of animals entering the ark, the beginning of the flood.
9	Noah's Ark	Noah's Ark - Part 3	40-60 seconds	Narrator: 'The floodwaters rose for forty days and nights. All living beings outside the ark perished. But Noah, his family, and the animals were safe inside.'	Visuals of the flood covering the earth, the ark floating safely.	The vast floodwaters, the ark, perished beings.	Noah - The protector. Family - Noah's supportive family. Animals - Safe in the ark.	Background Audio: Intense, dramatic instrumental music.	Dramatic instrumental music	Visual effects to enhance the flood coverage	Traditional Animation	The floodwaters rose for forty days and nights.'	Witness the power of God's judgment.	English	General Audience	Flood, Safety, Bible Stories	Experience the power of the flood and safety within the ark.	The floodwaters rise for forty days and nights, covering the earth and perishing all living beings outside the ark. Noah, his family, and the animals are safe inside.	Noah - The protector. Family - Noah's supportive family. Animals - Safe in the ark.	Visuals of the flood covering the earth, the ark floating safely.
10	Noah's Ark	Noah's Ark - Part 4	40-60 seconds	Narrator: 'After the flood, God made a covenant with Noah, promising never to destroy the earth with a flood again. He set a rainbow in the sky as a sign.'	Visuals of the ark resting on Mount Ararat, Noah's family exiting, the rainbow appearing.	The rainbow, Noah's family worshiping, the ark on dry ground.	God - The covenant maker. Noah - The faithful servant.	Background Audio: Uplifting, hopeful instrumental music.	Hopeful instrumental music	Visual effects to enhance the rainbow scene	Traditional Animation	God set a rainbow in the sky as a sign of His covenant.'	Celebrate God's promise and faithfulness.	English	General Audience	Covenant, Rainbow, Bible Stories	Celebrate God's promise with Noah after the flood.	After the flood, God makes a covenant with Noah, promising never to destroy the earth with a flood again, and sets a rainbow in the sky as a sign.	God - The covenant maker. Noah - The faithful servant.	Visuals of the ark resting on Mount Ararat, Noah's family exiting, the rainbow appearing.
11	Abraham's Call	The Call of Abraham - Part 1	40-60 seconds	Narrator: 'God called Abraham to leave his homeland and go to a land He would show him. Abraham obeyed, trusting in God's promise.'	Visuals of Abraham hearing God's call, preparing to leave home.	Abraham receiving God's call, packing to leave.	Abraham - A faithful servant of God, willing to obey.	Background Audio: Inspirational music	Uplifting instrumental music	Visual effects to highlight God's call	Traditional Animation	God called Abraham to leave his homeland.'	Witness Abraham's faith and obedience.	English	General Audience	Abraham, Call, Bible Stories	Discover the beginning of Abraham's journey of faith.	God calls Abraham to leave his homeland and go to a new land He will show him. Abraham obeys, demonstrating great faith.	Abraham receiving God's call, packing to leave, setting out.	Thumbnail of Abraham setting out on his journey
12	Abraham's Call	The Call of Abraham - Part 2	40-60 seconds	Narrator: 'Abraham journeyed to the land of Canaan. God promised to make him a great nation and bless all families of the earth through him.'	Visuals of Abraham traveling, arriving in Canaan.	Abraham traveling, the land of Canaan, God speaking.	Abraham - A faithful servant of God, journeying in faith.	Background Audio: Hopeful music	Hopeful instrumental music	Visual effects to highlight God's promise	Traditional Animation	God promised to make Abraham a great nation.'	See God's promise to Abraham.	English	General Audience	Abraham, Promise, Bible Stories	Follow Abraham's journey to Canaan and God's promise to him.	Abraham travels to Canaan, where God promises to make him a great nation and bless all families of the earth through him.	Abraham traveling through desert, reaching Canaan, God speaking.	Thumbnail of Abraham in Canaan
13	Sacrifice of Isaac	The Test of Abraham - Part 1	40-60 seconds	Narrator: 'God tested Abraham's faith by asking him to sacrifice his son Isaac. Abraham obeyed, trusting in God's plan.'	Visuals of God speaking to Abraham, preparing for sacrifice.	God speaking to Abraham, Abraham with Isaac.	Abraham - A faithful servant of God, tested in faith. Isaac - Abraham's beloved son.	Background Audio: Tense music	Dramatic instrumental music	Visual effects to highlight tension and drama	Traditional Animation	God tested Abraham's faith.'	Witness the ultimate test of faith.	English	General Audience	Abraham, Isaac, Sacrifice, Bible Stories	See how Abraham's faith is tested by God's command to sacrifice Isaac.	God tests Abraham's faith by asking him to sacrifice his son Isaac. Abraham obeys, showing his complete trust in God.	Abraham receiving God's command, preparing Isaac and the altar.	Thumbnail of Abraham and Isaac at the altar
14	Sacrifice of Isaac	The Test of Abraham - Part 2	40-60 seconds	Narrator: 'As Abraham was about to sacrifice Isaac, an angel stopped him. God provided a ram as a substitute sacrifice.'	Visuals of Abraham about to sacrifice Isaac, angel intervening.	Abraham with knife raised, the angel stopping him.	Abraham - A faithful servant of God. Isaac - Abraham's beloved son. Angel - God's messenger.	Background Audio: Dramatic to relief	Relief instrumental music	Visual effects to highlight divine intervention	Traditional Animation	God provided a ram as a substitute sacrifice.'	Experience the faithfulness of God.	English	General Audience	Abraham, Isaac, Sacrifice, Bible Stories	See how God intervenes in Abraham's test of faith and provides a ram.	As Abraham is about to sacrifice Isaac, an angel stops him, and God provides a ram as a substitute sacrifice.	Abraham with knife raised, angel stopping him, ram appearing.	Thumbnail of the angel stopping Abraham
15	Jacob's Ladder	Jacob's Ladder - Part 1	40-60 seconds	Narrator: 'Jacob had a dream of a ladder reaching to heaven with angels ascending and descending. God promised to be with him and bless him.'	Visuals of Jacob sleeping, ladder in dream, angels on ladder.	Jacob dreaming, ladder to heaven, angels.	Jacob - A patriarch of Israel, experiencing a divine vision.	Background Audio: Mystical music	Dreamlike instrumental music	Visual effects to enhance dream sequence	Traditional Animation	Jacob had a dream of a ladder reaching to heaven.'	Explore Jacob's divine dream.	English	General Audience	Jacob, Ladder, Dream, Bible Stories	Witness Jacob's dream of a ladder to heaven and God's promise to him.	Jacob dreams of a ladder reaching to heaven with angels ascending and descending. God promises to be with him and bless him.	Jacob lying down, the ladder appearing in his dream, angels.	Thumbnail of Jacob dreaming
16	Jacob's Ladder	Jacob's Ladder - Part 2	40-60 seconds	Narrator: 'Jacob set up a pillar in the place where God spoke to him. He called the place Bethel, meaning the house of God.'	Visuals of Jacob setting up a pillar, naming the place Bethel.	Jacob setting up a pillar, naming Bethel.	Jacob - A patriarch of Israel, experiencing a divine vision.	Background Audio: Reflective music	Reflective instrumental music	Visual effects to highlight Bethel	Traditional Animation	He called the place Bethel, meaning the house of God.'	Discover the significance of Bethel.	English	General Audience	Jacob, Bethel, Bible Stories	Learn about the significance of Bethel in Jacob's journey.	Jacob sets up a pillar in the place where God spoke to him, naming it Bethel, meaning the house of God.	Jacob setting up a pillar, naming the place Bethel.	Thumbnail of Jacob setting up a pillar
17	Joseph's Dreams	Joseph's Dreams - Part 1	40-60 seconds	Narrator: 'Joseph, the son of Jacob, had dreams of greatness which made his brothers jealous. They sold him into slavery in Egypt.'	Visuals of Joseph dreaming, his brothers' jealousy, being sold.	Joseph dreaming, brothers angry, Joseph sold.	Joseph - A dreamer and favored son of Jacob. Brothers - Jealous of Joseph.	Background Audio: Dramatic music	Dramatic instrumental music	Visual effects to highlight dreams and jealousy	Traditional Animation	Joseph had dreams of greatness.'	Discover Joseph's early trials.	English	General Audience	Joseph, Dreams, Bible Stories	Follow Joseph's dreams and the jealousy of his brothers.	Joseph, the son of Jacob, has dreams of greatness which make his brothers jealous. They sell him into slavery in Egypt.	Joseph dreaming, brothers' anger, Joseph sold into slavery.	Thumbnail of Joseph being sold
18	Joseph's Dreams	Joseph's Dreams - Part 2	40-60 seconds	Narrator: 'In Egypt, Joseph's ability to interpret dreams brought him to the attention of Pharaoh, who made him a ruler in Egypt.'	Visuals of Joseph interpreting dreams, becoming a ruler.	Joseph interpreting dreams, before Pharaoh, ruling.	Joseph - A dreamer and interpreter. Pharaoh - Ruler of Egypt.	Background Audio: Triumphant music	Triumphant instrumental music	Visual effects to highlight Joseph's rise	Traditional Animation	Joseph's ability to interpret dreams made him a ruler.'	See Joseph's rise to power in Egypt.	English	General Audience	Joseph, Pharaoh, Bible Stories	Witness Joseph's rise to power through his ability to interpret dreams.	In Egypt, Joseph's ability to interpret dreams brings him to the attention of Pharaoh, who makes him a ruler in Egypt.	Joseph interpreting dreams, before Pharaoh, ruling in Egypt.	Thumbnail of Joseph before Pharaoh
19	Moses' Birth	The Birth of Moses	40-60 seconds	Narrator: 'During a time when Hebrew babies were being killed, Moses' mother saved him by placing him in a basket and setting him afloat on the Nile.'	Visuals of Moses' mother placing him in a basket, setting him afloat.	Moses in a basket on the Nile, Pharaoh's daughter finding him.	Moses - Future leader of Israel, saved as a baby. Moses' Mother - Brave and protective. Pharaoh's Daughter - Compassionate rescuer.	Background Audio: Emotional music	Emotional instrumental music	Visual effects to enhance the Nile scene	Traditional Animation	Moses' mother saved him by placing him in a basket.'	Learn about Moses' miraculous survival.	English	General Audience	Moses, Birth, Bible Stories	Discover the story of Moses' birth and how he was saved.	During a time when Hebrew babies were being killed, Moses' mother saved him by placing him in a basket and setting him afloat on the Nile. Pharaoh's daughter found and adopted him.	Moses' mother placing him in the basket, setting him afloat, Pharaoh's daughter finding him.	Thumbnail of Moses in a basket on the Nile
20	Moses' Journey	Moses' Journey - Part 1	40-60 seconds	Narrator: 'Moses grew up in Pharaoh's palace but fled to the desert after killing an Egyptian who was beating a Hebrew slave.'	Visuals of Moses in the palace, witnessing the beating, fleeing.	Moses witnessing beating, killing Egyptian, fleeing.	Moses - Raised in Pharaoh's palace, destined for greatness. Egyptian - Oppressor. Hebrew Slave - Victim.	Background Audio: Dramatic music	Dramatic instrumental music	Visual effects to highlight the escape scene	Traditional Animation	Moses fled to the desert after killing an Egyptian.'	Follow Moses' early challenges.	English	General Audience	Moses, Egypt, Bible Stories	Follow Moses' journey from the palace to the desert.	Moses grew up in Pharaoh's palace but fled to the desert after killing an Egyptian who was beating a Hebrew slave. This marks the beginning of Moses' journey to become a leader.	Moses in the palace, witnessing the beating, killing Egyptian, fleeing to the desert.	Thumbnail of Moses fleeing to the desert""",
    "videos": [],      
    "audios": [], 
    "images": [[]],    
    "background_musics": [],          
    "subtitles": [],                
    "duration": 60,                  
    "font": "",                        
    "style": "",                      
    "theme": "",                 
    "audience": "",                
    "platform": [],  
    "total_output_videos": 2,
    "total_images_per_video": 5,
    "video_transiton": {"name":"generate_inward_vignette_transition_video"},
    "image_effects": {"name": effects},
}

def generate_prompts(prompt, number_of_prompts, user_series, prompt_type):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Create {number_of_prompts} similar but not identical prompts for {prompt_type}, each enhanced with more detail and creativity."},
            {"role": "user", "content": prompt}
        ]
    )

    prompts = response.choices[0].message.content.strip().split('\n\n')
   
    user_series[prompt_type].extend(prompts)

def generate_image_prompts(prompt, number_of_prompts, prompt_type, duration=60, image_time=5):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Create {number_of_prompts} similar but not identical prompts for {prompt_type}, each enhanced with more detail and creativity. There are time stamps so each image should be unique, I am creating a video of {duration} seconds so each image will be displaced for {image_time} seconds."},
            {"role": "user", "content": prompt}
        ]
    )

    prompts = response.choices[0].message.content.strip().split('\n')
    cleaned_list = list(filter(None, prompts))
    return cleaned_list

def generate_audio_scripts(video_prompts):
    for prompt in video_prompts:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Create a voiceover script for a video about: {prompt}. Ensure the output is just a voiceover paragraph. It should not exceed 400 characters, it should be maximum of 400 characters"},
                {"role": "user", "content": prompt}
            ]
        )

        audio_script = response.choices[0].message.content.strip()
        user_series["audios"].append(audio_script)

def generate_bgm_prompts(video_prompts):
    for prompt in video_prompts:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Create a background music prompt for a video about: {prompt}. Provide creative and suitable prompt that matched this video theme. Generate a prompt to pass to a background music generation tool. The prompt can include instruments and suitable music and starting and. Make it loopable. Ensure the output is just a prompt. It should not exceed 400 characters, it should be maximum of 400 characters"},
                {"role": "user", "content": prompt}
            ]
        )

        bgm_prompt = response.choices[0].message.content.strip()
        user_series["background_musics"].append(bgm_prompt)

def get_audio_duration(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_in_seconds = len(audio) / 1000  # Length of audio in seconds
        return duration_in_seconds

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def enhance_prompts(prompts):

    enhanced_prompts = {}

    for key, prompt in prompts.items():
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Enhance this prompt with more detail and creativity."},
                {"role": "user", "content": prompt}
            ]
        )

        enhanced_prompts[key] = response.choices[0].message.content

    return enhanced_prompts

def get_prompts(test_data):
    prompts = {
        "video_prompt": test_data["video"]["prompt"],
        "background_music": test_data["audio"]["prompt"],
        "voiceover_prompt": test_data["text_script"]["voiceover_prompt"]
    }
    return prompts

def generate_audio(text, output_file_path, model="tts-1", voice="alloy"):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    response.stream_to_file(Path(output_file_path))

def transcribe_audio(file_path, model="whisper-1"):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file
        )
    return transcription.text

def match_durations(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    min_duration = min(video.duration, audio.duration)

    video = video.subclip(0, min_duration)
    audio = audio.subclip(0, min_duration)

    video = video.set_audio(audio)

    video.write_videofile(output_path, codec='libx264', audio_codec='aac')

def generate_subtitle_clips(data, video):
    if data is None or 'words' not in data or 'word_start_times' not in data or 'word_end_times' not in data:
        raise ValueError("Invalid or missing data structure for subtitles.")
    
    clips = []
    font_style = random.choice(list(text_styles.values()))
    y_position = video.size[1] * 0.75
    
    for word, start, end in zip(data['words'], data['word_start_times'], data['word_end_times']):
        text_clip = TextClip(
            txt=word,
            size=font_style['size'],
            color=font_style['color'], 
            fontsize=font_style['fontsize'],
            bg_color=font_style['bg_color'],
            font=font_style['font'],
            stroke_color=font_style['stroke_color'],
            stroke_width=font_style['stroke_width'],
            method=font_style['method'],
            kerning=font_style['kerning'],
            align=font_style['align'],
            interline=font_style['interline'],
            transparent=font_style['transparent'],
            remove_temp=font_style['remove_temp'],
            print_cmd=font_style['print_cmd']
        ).set_position(('center', y_position)).set_start(start).set_duration(end - start)

        clips.append(text_clip)
    return clips

def add_watermark(video_path, output_path, audio_path, bgm_path, subtitle_data, text=None, image_path=None, position=('center', 'center'), 
                  font='Arial', font_size=24, color='white', opacity=0.5, padding=10, bgm_volume=0.5):
    """
    Add a watermark to a video.
    
    :param video_path: Path to the input video file.
    :param output_path: Path to save the output video file.
    :param text: Text to use as a watermark (optional).
    :param image_path: Path to an image to use as a watermark (optional).
    :param position: Position of the watermark in the video ('center', 'center' by default).
    :param font: Font type for the text watermark.
    :param font_size: Font size for the text watermark.
    :param color: Font color for the text watermark.
    :param opacity: Opacity level for the watermark (0 to 1).
    """
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    bgm_audio = AudioFileClip(bgm_path)
    
    min_duration = min(video.duration, audio.duration)

    video = video.subclip(0, min_duration)
    audio = audio.subclip(0, min_duration)

    bgm_audio = afx.volumex(bgm_audio, bgm_volume)

    video = video.set_audio(None)
    
    bgm_audio = afx.audio_loop(bgm_audio, duration=audio.duration)

    combined_audio = CompositeAudioClip([bgm_audio.set_duration(video.duration), audio.set_duration(video.duration)])

    video = video.set_audio(combined_audio)

    if text:
        watermark = (TextClip(text, fontsize=font_size, font=font, color=color)
                     .set_opacity(opacity)
                     .set_pos(position)
                     .set_duration(video.duration)
                     .margin(left=padding, right=padding, top=padding, bottom=padding, opacity=0))
    elif image_path:
        watermark = (ImageClip(image_path)
                     .set_duration(video.duration)
                     .set_opacity(opacity)
                     .set_pos(position)
                     .margin(left=padding, right=padding, top=padding, bottom=padding, opacity=0))
    else:
        raise ValueError("Either text or image_path must be provided for the watermark.")

    subtitle_clips = generate_subtitle_clips(subtitle_data, video)

    # Composite the video with the watermark
    final_video = CompositeVideoClip([video, watermark] + subtitle_clips)

    # Write the result to a file
    final_video.write_videofile(output_path, codec='libx264', fps=video.fps, audio_codec='aac')

def create_video_with_audio_and_text(video_path, audio_path, bgm_path, output_path, subtitle_data, text=None, image_path=None, position=('center', 'center'), 
                  font='Arial', font_size=24, color='white', opacity=0.5, padding=10, bgm_volume=0.5):
   
    # Load the video and audio files
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    bgm_audio = AudioFileClip(bgm_path)

    min_duration = min(video.duration, audio.duration)

    video = video.subclip(0, min_duration)
    audio = audio.subclip(0, min_duration)

    bgm_audio = afx.volumex(bgm_audio, bgm_volume)

    video = video.set_audio(None)
    
    bgm_audio = afx.audio_loop(bgm_audio, duration=audio.duration)

    combined_audio = CompositeAudioClip([bgm_audio.set_duration(video.duration), audio.set_duration(video.duration)])

    video = video.set_audio(combined_audio)

    if text:
        watermark = (TextClip(text, fontsize=font_size, font=font, color=color)
                     .set_opacity(opacity)
                     .set_pos(position)
                     .set_duration(video.duration)
                      .margin(left=padding, right=padding, top=padding, bottom=padding, opacity=0))
    elif image_path:
        watermark = (ImageClip(image_path)
                     .set_duration(video.duration)
                     .set_opacity(opacity)
                     .set_pos(position)
                      .margin(left=padding, right=padding, top=padding, bottom=padding, opacity=0))
        
        
    subtitle_clips = generate_subtitle_clips(subtitle_data, video)

    # Create the final composite video clip with the watermark
    final = CompositeVideoClip([video, watermark] + subtitle_clips)

    # Export the final video
    final.write_videofile(output_path, codec='libx264', audio_codec='aac')

# Generae the prompt for series
generate_prompts(user_series["user_prompt"], user_series["total_output_videos"], user_series, "videos")

generate_audio_scripts(user_series["videos"])

generate_bgm_prompts(user_series["videos"])

# audio_prompts.split_transcript(user_series["audios"][0], user_series, "audios")

# # Audio and BGM
bgm_path = f"/Users/toheed/PanduAI/backend/workflow/bgm/bgm{filenumber}.mp3"
audio_path = f"/Users/toheed/PanduAI/backend/workflow/audio/audio{filenumber}.mp3"

subtitle_data = speech_synthesis.generate_tts_with_timestamps(user_series["audios"][0], audio_path)
bgm_data = speech_synthesis.generate_and_save_sound(user_series["background_musics"][0], bgm_path)

# Effects and Images
image_prompts = user_series["audios"]

# image_prompts = generate_image_prompts(user_series["user_prompt"], user_series["total_images_per_video"], "images", user_series["duration"], 5)
print(image_prompts)

api_key = STABILITY_SECRET_KEY
negative_prompt = 'fire'
aspect_ratio = '9:16'
seed = 12345
output_format = 'jpeg'
output_folder = '/Users/toheed/PanduAI/backend/Images'
input_video_folder = '/Users/toheed/PanduAI/backend/Videos'
# # output_video_path = f"/Users/toheed/PanduAI/backend/workflow/video/stitched_video{filenumber}.mp4"
output_video_path = '/Users/toheed/PanduAI/backend/workflow/video/'
input_image_folder = '/Users/toheed/PanduAI/backend/Images'
output_video_folder = '/Users/toheed/PanduAI/backend/Videos'
output_filename_stiched = f"/Users/toheed/PanduAI/backend/workflow/video/stitched_video{filenumber}.mp4"

# folder_path = "/Users/toheed/PanduAI/backend/output"
# durations = audio_prompts.get_durations_from_folder(folder_path)
# totalimages = [round(i / 5) for i in durations]

# Generate images
audio_prompts.generate_multiple_images(api_key, image_prompts, output_dir=output_folder)

# Effects
image_effects.process_multiple_images(input_image_folder, output_video_folder)

# Transition
image_effects.stitch_videos_with_random_transition(input_video_folder, output_filename_stiched)

output_path= f"/Users/toheed/PanduAI/backend/workflow/result/result_final_with_watermark_{filenumber}.mp4"
video_path= f"/Users/toheed/PanduAI/backend/workflow/video/stitched_video{filenumber}.mp4"

# Compile everything
create_video_with_audio_and_text(
    video_path= video_path,
    audio_path=f"/Users/toheed/PanduAI/backend/workflow/audio/audio{filenumber}.mp3",
    bgm_path=f"/Users/toheed/PanduAI/backend/workflow/bgm/bgm{filenumber}.mp3",
    output_path= output_path,
    subtitle_data=subtitle_data,
    text="Pandu AI", 
    position=("right", "top"), 
    font="Helvetica-Bold", 
    font_size=50, 
    color="gray", 
    opacity=0.8,
    padding=10,
    bgm_volume=0.15
)

# # add_watermark(
# #     video_path, 
# #     output_path,   
# #     audio_path=f"/Users/toheed/PanduAI/backend/workflow/audio/audio{filenumber}.mp3",
# #     bgm_path=f"/Users/toheed/PanduAI/backend/workflow/bgm/bgm{filenumber}.mp3",
# #     subtitle_data=subtitle_data, 
# #     text="Pandu AI", 
# #     position=("top", "right"), 
# #     font="Helvetica-Bold", 
# #     font_size=50, 
# #     color="gray", 
# #     opacity=0.8,
# #     bgm_volume=0.15
# # )
# # add_watermark(video_path, output_path, image_path="watermark.png", position=("left", "top"), opacity=0.5)
