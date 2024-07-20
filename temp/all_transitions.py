def generate_wipe_top_to_bottom(video1, video2, frames=7, fps=30, frame_repeat=1):
    clip1 = VideoFileClip(video1)
    clip2 = VideoFileClip(video2)

    width, height = clip1.size

    def make_frame(t):
        frame1 = clip1.get_frame(t)
        frame2 = clip2.get_frame(t)

        i = int((t * fps) / frame_repeat) % frames
        y = int(height * (i / frames))

        wipe_frame = np.zeros_like(frame1)
        wipe_frame[:height-y, :] = frame1[y:, :]
        wipe_frame[height-y:, :] = frame2[:y, :]

        return wipe_frame

    duration = max(clip1.duration, clip2.duration)
    wipe_clip = VideoClip(make_frame, duration=duration).set_fps(fps)

    return wipe_clip

def generate_wipe_bottom_to_top(video1, video2, output_video_path, frames=7, fps=30, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        for _ in range(frame_repeat):
            out.write(frame1)

    if not cap2.isOpened():
        print("Error: Unable to open the second video.")
        return

    for i in range(frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break

        if ret1:
            frame1 = cv2.resize(frame1, (width, height))
        else:
            frame1 = np.zeros((height, width, 3), dtype=np.uint8)

        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
        else:
            frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        y = int(height * (i / frames))
        wipe_frame = np.zeros_like(frame1)
        wipe_frame[y:, :] = frame1[:height-y, :]
        wipe_frame[:y, :] = frame2[height-y:, :]

        for _ in range(frame_repeat):
            out.write(wipe_frame)

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()

def generate_wipe_left_to_right(video1, video2, output_video_path, frames=7, fps=30, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        for _ in range(frame_repeat):
            out.write(frame1)

    if not cap2.isOpened():
        print("Error: Unable to open the second video.")
        return

    for i in range(frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break

        if ret1:
            frame1 = cv2.resize(frame1, (width, height))
        else:
            frame1 = np.zeros((height, width, 3), dtype=np.uint8)

        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
        else:
            frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        x = int(width * (i / frames))
        wipe_frame = np.zeros_like(frame1)
        wipe_frame[:, :x] = frame2[:, :x]
        wipe_frame[:, x:] = frame1[:, :width-x]

        for _ in range(frame_repeat):
            out.write(wipe_frame)

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()

def generate_wipe_right_to_left(video1, video2, output_video_path, frames=7, fps=30, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        for _ in range(frame_repeat):
            out.write(frame1)

    if not cap2.isOpened():
        print("Error: Unable to open the second video.")
        return

    for i in range(frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break

        if ret1:
            frame1 = cv2.resize(frame1, (width, height))
        else:
            frame1 = np.zeros((height, width, 3), dtype=np.uint8)

        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
        else:
            frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        x = int(width * (i / frames))
        wipe_frame = np.zeros_like(frame1)
        wipe_frame[:, :width-x] = frame1[:, x:]
        wipe_frame[:, width-x:] = frame2[:, :x]

        for _ in range(frame_repeat):
            out.write(wipe_frame)

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()

def generate_horizontal_stripes(video1, video2, transition_frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Apply the horizontal stripes transition
    num_stripes = 8
    stripe_height = height // num_stripes
    transition_step = max(1, transition_frames // num_stripes)

    for frame_idx in range(transition_frames):
        # Create a mask
        mask = np.zeros((height, width), dtype=np.uint8)

        stripes_to_reveal = frame_idx // transition_step
        for y in range(stripes_to_reveal):
            mask[y * stripe_height:(y + 1) * stripe_height, :] = 255

        mask = cv2.merge([mask, mask, mask])

        img1_part = cv2.bitwise_and(last_frame_first_video, cv2.bitwise_not(mask))
        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_vertical_stripes(video1, video2, transition_frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Apply the vertical stripes transition
    num_stripes = 8
    stripe_width = width // num_stripes
    transition_step = max(1, transition_frames // num_stripes)

    for frame_idx in range(transition_frames):
        # Create a mask
        mask = np.zeros((height, width), dtype=np.uint8)

        stripes_to_reveal = frame_idx // transition_step
        for x in range(stripes_to_reveal):
            mask[:, x * stripe_width:(x + 1) * stripe_width] = 255

        mask = cv2.merge([mask, mask, mask])

        img1_part = cv2.bitwise_and(last_frame_first_video, cv2.bitwise_not(mask))
        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_box_inward(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the inward box transition effect
    for i in range(frames):
        # Calculate the size of the revealing box
        box_size = int((1 - (i / frames)) * width)  # Width and height decrease
        x_start = (width - box_size) // 2
        y_start = (height - box_size) // 2

        # Create a mask with a rectangle
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y_start:y_start + box_size, x_start:x_start + box_size] = 255
        mask = cv2.merge([mask, mask, mask])  # Ensure mask has 3 channels

        img1_part = cv2.bitwise_and(last_frame_first_video, cv2.bitwise_not(mask))
        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_box_outward(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the outward box transition effect
    for i in range(frames):
        # Calculate the size of the revealing box
        box_size = int((i / frames) * width)  # Width and height increase
        x_start = (width - box_size) // 2
        y_start = (height - box_size) // 2

        # Create a mask with a rectangle
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y_start:y_start + box_size, x_start:x_start + box_size] = 255
        mask = cv2.merge([mask, mask, mask])  # Ensure mask has 3 channels

        img1_part = cv2.bitwise_and(last_frame_first_video, cv2.bitwise_not(mask))
        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_horizontal_sliding_door(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the horizontal sliding door transition effect
    for i in range(frames):
        offset = int((i / frames) * (width // 2))

        left_part = last_frame_first_video[:, :width // 2 - offset]
        right_part = last_frame_first_video[:, width // 2 + offset:]
        middle_part = first_frame_second_video_resized[:, width // 2 - offset:width // 2 + offset]

        combined_frame = np.hstack((left_part, middle_part, right_part))

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_vertical_sliding_door(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the vertical sliding door transition effect
    for i in range(frames):
        offset = int((i / frames) * (height // 2))

        top_part = last_frame_first_video[:height // 2 - offset, :]
        bottom_part = last_frame_first_video[height // 2 + offset:, :]
        middle_part = first_frame_second_video_resized[height // 2 - offset:height // 2 + offset, :]

        combined_frame = np.vstack((top_part, middle_part, bottom_part))

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_diagonal_sliding_door_tl_br(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the diagonal sliding door transition effect
    for i in range(frames):
        offset_x = int((i / frames) * (width // 2))
        offset_y = int((i / frames) * (height // 2))

        mask = np.zeros_like(last_frame_first_video)

        # Top-left part
        mask[:height // 2 - offset_y, :width // 2 - offset_x] = last_frame_first_video[:height // 2 - offset_y, :width // 2 - offset_x]
        
        # Bottom-right part
        mask[height // 2 + offset_y:, width // 2 + offset_x:] = last_frame_first_video[height // 2 + offset_y:, width // 2 + offset_x:]

        # Middle part
        mask[height // 2 - offset_y:height // 2 + offset_y, width // 2 - offset_x:width // 2 + offset_x] = first_frame_second_video_resized[height // 2 - offset_y:height // 2 + offset_y, width // 2 - offset_x:width // 2 + offset_x]

        for _ in range(frame_repeat):
            frames_list.append(mask)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_diagonal_sliding_door_bl_tr(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the diagonal sliding door transition effect
    for i in range(frames):
        offset_x = int((i / frames) * (width // 2))
        offset_y = int((i / frames) * (height // 2))

        mask = np.zeros_like(last_frame_first_video)

        # Bottom-left part
        mask[height // 2 + offset_y:, :width // 2 - offset_x] = last_frame_first_video[height // 2 + offset_y:, :width // 2 - offset_x]
        
        # Top-right part
        mask[:height // 2 - offset_y, width // 2 + offset_x:] = last_frame_first_video[:height // 2 - offset_y, width // 2 + offset_x:]

        # Middle part
        mask[height // 2 - offset_y:height // 2 + offset_y, width // 2 - offset_x:width // 2 + offset_x] = first_frame_second_video_resized[height // 2 - offset_y:height // 2 + offset_y, width // 2 - offset_x:width // 2 + offset_x]

        for _ in range(frame_repeat):
            frames_list.append(mask)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_minimize_to_topleft(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the minimizing to top-left transition effect
    for i in range(frames):
        scale = 1 - (i / frames)
        scaled_frame = cv2.resize(last_frame_first_video, (int(width * scale), int(height * scale)))

        mask = np.zeros_like(last_frame_first_video)
        mask[:int(height * scale), :int(width * scale)] = scaled_frame

        combined_frame = np.where(mask == 0, first_frame_second_video_resized, mask).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_minimize_to_topright(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the minimizing to top-right transition effect
    for i in range(frames):
        scale = 1 - (i / frames)
        scaled_frame = cv2.resize(last_frame_first_video, (int(width * scale), int(height * scale)))

        mask = np.zeros_like(last_frame_first_video)
        mask[:int(height * scale), width - int(width * scale):] = scaled_frame

        combined_frame = np.where(mask == 0, first_frame_second_video_resized, mask).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_minimize_to_bottomleft(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the minimizing to bottom-left transition effect
    for i in range(frames):
        scale = 1 - (i / frames)
        scaled_frame = cv2.resize(last_frame_first_video, (int(width * scale), int(height * scale)))

        mask = np.zeros_like(last_frame_first_video)
        mask[height - int(height * scale):, :int(width * scale)] = scaled_frame

        combined_frame = np.where(mask == 0, first_frame_second_video_resized, mask).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_minimize_to_bottomright(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the minimizing to bottom-right transition effect
    for i in range(frames):
        scale = 1 - (i / frames)
        scaled_frame = cv2.resize(last_frame_first_video, (int(width * scale), int(height * scale)))

        mask = np.zeros_like(last_frame_first_video)
        mask[height - int(height * scale):, width - int(width * scale):] = scaled_frame

        combined_frame = np.where(mask == 0, first_frame_second_video_resized, mask).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_halfstripe_horizontal(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the two stripe horizontal transition effect
    for i in range(frames):
        offset = int(width * (i / frames))
        top_half = last_frame_first_video[:height // 2, :]
        bottom_half = last_frame_first_video[height // 2:, :]
        
        top_frame = np.zeros_like(first_frame_second_video_resized)
        bottom_frame = np.zeros_like(first_frame_second_video_resized)
        
        top_frame[:height // 2, :width - offset] = first_frame_second_video_resized[:height // 2, offset:]
        bottom_frame[height // 2:, :width - offset] = first_frame_second_video_resized[height // 2:, offset:]

        combined_frame = top_frame + bottom_frame

        combined_frame = np.where(combined_frame == 0, last_frame_first_video, combined_frame).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_halfstripe_vertical(video1, video2, frames=30, fps=30, frame_repeat=1):
    # Capture both video files
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    # Get the width and height of the videos
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture all frames from the first video
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the two stripe vertical transition effect
    for i in range(frames):
        offset = int(height * (i / frames))
        left_half = last_frame_first_video[:, :width // 2]
        right_half = last_frame_first_video[:, width // 2:]
        
        left_frame = np.zeros_like(first_frame_second_video_resized)
        right_frame = np.zeros_like(first_frame_second_video_resized)
        
        left_frame[:height - offset, :width // 2] = first_frame_second_video_resized[offset:, :width // 2]
        right_frame[:height - offset, width // 2:] = first_frame_second_video_resized[offset:, width // 2:]

        combined_frame = np.hstack((left_frame, right_frame))

        combined_frame = np.where(combined_frame == 0, last_frame_first_video, combined_frame).astype(np.uint8)

        for _ in range(frame_repeat):
            frames_list.append(combined_frame)

    # Capture all frames from the second video
    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frame2_resized = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2_resized)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_outward_vignette_transition(video1, video2, frames=18, fps=30, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture the last frame of the first video
    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the outward vignette transition frames
    for i in range(frames):
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (width//2, height//2), int((width//2) * (i+1) / frames), 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = cv2.merge([mask, mask, mask])

        img1_part = cv2.bitwise_and(last_frame_first_video, mask)
        inverted_mask = cv2.bitwise_not(mask)
        img2_part = cv2.bitwise_and(first_frame_second_video_resized, inverted_mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Add all frames from the second video
    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip

def generate_inward_vignette_transition(video1, video2, frames=18, fps=30, frame_repeat=1):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    # Capture the last frame of the first video
    while True:
        ret, frame1 = cap1.read()
        if not ret:
            break
        frame1_resized = cv2.resize(frame1, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame1_resized)
    last_frame_first_video = frame1_resized

    # Capture the first frame of the second video
    ret, first_frame_second_video = cap2.read()
    if not ret:
        print("Error: Unable to read the second video.")
        return None
    first_frame_second_video_resized = cv2.resize(first_frame_second_video, (width, height))

    # Create the inward vignette transition frames
    for i in range(frames):
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (width//2, height//2), int((width//2) * (frames-i) / frames), 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = cv2.merge([mask, mask, mask])

        img2_part = cv2.bitwise_and(first_frame_second_video_resized, mask)
        inverted_mask = cv2.bitwise_not(mask)
        img1_part = cv2.bitwise_and(last_frame_first_video, inverted_mask)
        blended_image = cv2.add(img1_part, img2_part)

        for _ in range(frame_repeat):
            frames_list.append(blended_image)

    # Add all frames from the second video
    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (width, height))
        for _ in range(frame_repeat):
            frames_list.append(frame2)

    cap1.release()
    cap2.release()

    video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_list], fps=fps)
    return video_clip