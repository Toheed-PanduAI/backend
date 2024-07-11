from passlib.context import CryptContext
# import boto3
# from botocore.exceptions import NoCredentialsError
from pathlib import Path

# def upload_to_aws(file_path, bucket_name, object_name):
#     # Initialize the S3 client
#     s3 = boto3.client('s3')

#     try:
#         # Upload the file to S3
#         s3.upload_file(file_path, bucket_name, object_name)
#         print(f"File uploaded successfully to {bucket_name}/{object_name}")
#         return True

#     except NoCredentialsError:
#         print("AWS credentials not available.")
#         return False

# # Example usage:
# def upload_files_to_aws(video_path, audio_path, bgm_path, bucket_name):
#     # Upload video file
#     video_file = Path(video_path)
#     if video_file.is_file():
#         upload_to_aws(video_path, bucket_name, video_file.name)
#     else:
#         print(f"Video file {video_path} not found.")

#     # Upload audio file
#     audio_file = Path(audio_path)
#     if audio_file.is_file():
#         upload_to_aws(audio_path, bucket_name, audio_file.name)
#     else:
#         print(f"Audio file {audio_path} not found.")

#     # Upload background music file
#     bgm_file = Path(bgm_path)
#     if bgm_file.is_file():
#         upload_to_aws(bgm_path, bucket_name, bgm_file.name)
#     else:
#         print(f"BGM file {bgm_path} not found.")


def characters_to_words(characters, start_times, end_times):
    # Ensure the lengths of the lists match
    if len(characters) != len(start_times) or len(characters) != len(end_times):
        print(len(characters), len(start_times), len(end_times))
        raise ValueError("Input lists must have the same length.")

    words = []
    word_start_times = []
    word_end_times = []

    word = []
    word_start_time = None
    word_end_time = None

    for i, char in enumerate(characters):
        if char != ' ':
            if not word:
                word_start_time = start_times[i]
            word.append(char)
            word_end_time = end_times[i]
        else:
            if word:
                words.append(''.join(word))
                word_start_times.append(word_start_time)
                word_end_times.append(word_end_time)
                word = []
                word_start_time = None
                word_end_time = None

    # Add the last word if there is one
    if word:
        words.append(''.join(word))
        word_start_times.append(word_start_time)
        word_end_times.append(word_end_time)

    return {
        'words': words,
        'word_start_times': word_start_times,
        'word_end_times': word_end_times
    }

text_styles = {
    1: {
        'size': None,
        'color': 'MidnightBlue',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': None,
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    2: {
        'size': None,
        'color': 'MidnightBlue',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'black',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    3: {
        'size': None,
        'color': 'MintCream',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'gray',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    4: {
        'size': None,
        'color': 'brown2',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    5: {
        'size': None,
        'color': 'brown2',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    6: {
        'size': None,
        'color': 'DarkSlateGrey',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'gray',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    7: {
        'size': None,
        'color': 'black',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'black',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    8: {
        'size': None,
        'color': 'DarkSlateGrey',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'MintCream',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    9: {
        'size': None,
        'color': 'DarkGray',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Futura-Bold',
        'stroke_color': 'beige',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    10: {
        'size': None,
        'color': 'DarkSlateGrey',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Futura-Bold',
        'stroke_color': 'beige',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    11: {
        'size': None,
        'color': 'DarkGray',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Futura-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    12: {
        'size': None,
        'color': 'white',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'DarkSlateGrey',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    13: {
        'size': None,
        'color': 'black',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    14: {
        'size': None,
        'color': 'white',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'beige',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    15: {
        'size': None,
        'color': 'MidnightBlue',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'black',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    16: {
        'size': None,
        'color': 'brown2',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    17: {
        'size': None,
        'color': 'Black',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Helvetica-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    18: {
        'size': None,
        'color': 'DarkGray',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'MintCream',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    19: {
        'size': None,
        'color': 'DarkSlateGrey',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    20: {
        'size': None,
        'color': 'DarkSlateGrey',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'MintCream',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    21: {
        'size': None,
        'color': 'black',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'black',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    22: {
        'size': None,
        'color': 'Black',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'beige',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    23: {
        'size': None,
        'color': 'MintCream',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    24: {
        'size': None,
        'color': 'DarkSlateGrey',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Georgia-Bold',
        'stroke_color': 'black',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    25: {
        'size': None,
        'color': 'beige',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Futura-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    26: {
        'size': None,
        'color': 'white',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Futura-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    27: {
        'size': None,
        'color': 'DarkSlateGrey',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Futura-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    28: {
        'size': None,
        'color': 'black',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Futura-Bold',
        'stroke_color': 'white',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    29: {
        'size': None,
        'color': 'beige',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Futura-Bold',
        'stroke_color': 'MintCream',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    },
    30: {
        'size': None,
        'color': 'MintCream',
        'bg_color': 'transparent',
        'fontsize': 75,
        'font': 'Futura-Bold',
        'stroke_color': 'beige',
        'stroke_width': 6,
        'method': 'caption',
        'kerning': None,
        'align': 'center',
        'interline': None,
        'transparent': True,
        'remove_temp': True,
        'print_cmd': False
    }
}


effect_images = {
    1: "generate_pan_bottom_to_top_video",
    2: "generate_pan_top_right_to_bottom_left_video",
    3: "generate_pan_bottom_left_to_top_right_video",
    4: "generate_pan_bottom_right_to_top_left_video",
    5: "generate_pan_top_left_to_bottom_right_video",
    6: "generate_pan_right_to_left_video",
    7: "generate_pan_top_to_bottom_video",
    8: "generate_pan_left_to_right_video"
}

transition_images = {
    1: "crossfade_transition",
    2: "fadeinout_transition",
    3: "slide_transition",
    # 4: "generate_outward_vignette_transition_video",
    # 5: "generate_inward_vignette_transition_video",
    # 6: "generate_wipe_bottom_to_top",
    # 7: "generate_wipe_top_to_bottom",
    # 8: "generate_wipe_left_to_right",
    # 9: "generate_wipe_right_to_left"
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)
