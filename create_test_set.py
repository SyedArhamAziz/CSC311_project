import pandas as pd
import numpy as np

train_df = pd.read_csv("ml_challenge_dataset_fixed.csv")
cols = list(train_df.columns)
# cols: unique_id, Painting, [2]=intensity, [3]=describe, [4]=sombre,
#       [5]=content, [6]=calm, [7]=uneasy, [8]=colours, [9]=objects,
#       [10]=price, [11]=room, [12]=who, [13]=season, [14]=food, [15]=soundtrack

POM  = "The Persistence of Memory"
SN   = "The Starry Night"
WLP  = "The Water Lily Pond"

rows = [
    # --- The Persistence of Memory ---
    ("test_001", POM, 8, "The melting clocks evoke a surreal sense of time dissolving into nothing. Dreamlike and unsettling.", "5 - Strongly agree", "1 - Strongly disagree", "2 - Disagree", "4 - Agree", 3, 5, 500, "Office", "By yourself", "Summer", "Tiramisu", "A slow eerie ambient drone with distant echoes that stretches uncomfortably long"),
    ("test_002", POM, 7, "Time feels suspended and strange. A quiet dread like something is deeply wrong.", "4 - Agree", "2 - Disagree", "2 - Disagree", "4 - Agree", 4, 4, 200, "Bedroom", "Friends", "Fall", "Overripe banana bread", "A minimalist haunting melody on strings with long pauses and dissonant chords"),
    ("test_003", POM, 9, "Deep philosophical unease. The distorted clocks make me question reality and time itself.", "5 - Strongly agree", "1 - Strongly disagree", "1 - Strongly disagree", "5 - Strongly agree", 3, 6, 1000, "Office", "By yourself", "Winter", "Melted ice cream", "A deep orchestral score with warped tempo that speeds and slows unpredictably"),
    ("test_004", POM, 6, "Deeply sad. The barren landscape and drooping clocks feel so lonely and abandoned.", "4 - Agree", "2 - Disagree", "3 - Neutral/Unsure", "3 - Neutral/Unsure", 4, 4, 300, "Living room", "By yourself", "Fall", "Dark chocolate", "Sparse piano notes in minor key with long reverb and a distant wind softly whistling"),
    ("test_005", POM, 8, "Time feels meaningless. Everything feels heavy and slow and deeply out of place.", "5 - Strongly agree", "1 - Strongly disagree", "2 - Disagree", "4 - Agree", 3, 5, 750, "Office", "By yourself", "Summer", "Cold coffee", "An ambient soundscape with slow unsettling tones shifting in and out like a dream"),
    ("test_006", POM, 7, "Surreal and thought-provoking. Like standing in a memory that no longer makes sense.", "4 - Agree", "2 - Disagree", "2 - Disagree", "4 - Agree", 3, 5, 400, "Bedroom", "By yourself", "Fall", "Stale croissant", "A slow haunting guitar piece layered with distant voices and unusual instrument tones"),
    ("test_007", POM, 9, "Deeply disturbing but fascinating. Time feels like it is decaying before my eyes.", "5 - Strongly agree", "1 - Strongly disagree", "1 - Strongly disagree", "5 - Strongly agree", 4, 6, 2000, "Office", "By yourself", "Winter", "Expired cheese", "A dark electronic ambient track with industrial rhythms and long sustained notes"),
    ("test_008", POM, 6, "Disoriented and reflective. The unusual landscape draws me into a meditative trance.", "3 - Neutral/Unsure", "2 - Disagree", "3 - Neutral/Unsure", "3 - Neutral/Unsure", 4, 4, 250, "Bedroom", "By yourself", "Fall", "Bittersweet marmalade", "A melancholic cello solo with distant ambient sound and very quiet breathing"),
    ("test_009", POM, 8, "Existential dread and curiosity. Like the world has stopped but is somehow still moving.", "5 - Strongly agree", "1 - Strongly disagree", "2 - Disagree", "4 - Agree", 3, 5, 600, "Office", "By yourself", "Winter", "Warm stale soup", "A minimalist orchestral piece with slow swells like time compressing and expanding"),
    ("test_010", POM, 7, "Dreamlike and vaguely sad. Time is slipping through my fingers and I cannot stop it.", "4 - Agree", "2 - Disagree", "2 - Disagree", "4 - Agree", 3, 4, 350, "Bedroom", "By yourself", "Fall", "Overcooked pasta", "A soft ethereal hum with slow arpeggiated strings and a sense of fading distance"),

    # --- The Starry Night ---
    ("test_011", SN, 8, "Wonder and awe. The swirling sky is dynamic and full of energy yet deeply peaceful.", "2 - Disagree", "4 - Agree", "4 - Agree", "2 - Disagree", 5, 7, 800, "Living room", "Friends", "Summer", "Blueberry pie", "A sweeping orchestral piece with rising strings and hopeful warm brass instruments"),
    ("test_012", SN, 7, "Bold swirling brushstrokes give a sense of movement and life. I feel hopeful and inspired.", "2 - Disagree", "4 - Agree", "4 - Agree", "1 - Strongly disagree", 5, 6, 500, "Living room", "Family members", "Spring", "Lavender macaron", "A passionate romantic symphony with sweeping melodies and a warm glowing tone"),
    ("test_013", SN, 9, "Overwhelming beauty. Luminous stars and rolling sky feel alive. Small but deeply connected.", "1 - Strongly disagree", "5 - Strongly agree", "4 - Agree", "1 - Strongly disagree", 6, 8, 2000, "Living room", "Friends", "Summer", "Cotton candy", "A grand cinematic score with soaring strings and triumphant horns building to a crescendo"),
    ("test_014", SN, 6, "Peaceful and dreamy. Glowing stars are comforting, like cool nights spent looking at the sky.", "2 - Disagree", "4 - Agree", "5 - Strongly agree", "1 - Strongly disagree", 5, 6, 400, "Bedroom", "Family members", "Summer", "Chamomile cake", "A gentle flowing melody with soft strings and a lilting tempo that feels like floating gently"),
    ("test_015", SN, 8, "Energetic and emotional. The swirling sky channels turbulent emotions into something beautiful.", "2 - Disagree", "3 - Neutral/Unsure", "3 - Neutral/Unsure", "2 - Disagree", 5, 7, 700, "Living room", "Friends", "Spring", "Galaxy chocolate bar", "A dramatic orchestral piece that alternates between intense turbulence and quiet peaceful beauty"),
    ("test_016", SN, 7, "A sense of motion and wonder. The night sky swirls with life. Both anxious and beautiful.", "3 - Neutral/Unsure", "3 - Neutral/Unsure", "3 - Neutral/Unsure", "2 - Disagree", 5, 6, 450, "Living room", "Friends", "Summer", "Grape sorbet", "A spiralling piano piece that builds in intensity before resolving into peaceful harmonic chords"),
    ("test_017", SN, 9, "Awe-inspiring. The painting captures energy of a universe in motion. I feel joy and wonder.", "1 - Strongly disagree", "5 - Strongly agree", "4 - Agree", "1 - Strongly disagree", 6, 8, 3000, "Living room", "Friends", "Spring", "Starfruit tart", "A triumphant orchestral work moving from delicate pizzicato strings to full powerful ensemble"),
    ("test_018", SN, 6, "Calm and energetic together. Like a meditation under the open sky at night. Spirals peacefully.", "2 - Disagree", "4 - Agree", "5 - Strongly agree", "1 - Strongly disagree", 4, 5, 300, "Bedroom", "By yourself", "Summer", "Honey glazed pastry", "A slow ambient nocturne with soft high piano notes and a gently pulsing harmonic background"),
    ("test_019", SN, 8, "I feel uplifted and moved. Deep blues and swirling forms are emotionally powerful and freeing.", "2 - Disagree", "4 - Agree", "4 - Agree", "1 - Strongly disagree", 5, 7, 900, "Living room", "Friends", "Spring", "Blueberry cheesecake", "A soaring romantic piece with lush strings and a melody that spirals upward with emotion"),
    ("test_020", SN, 7, "The painting feels alive and breathing. Quiet joy and connection to something far bigger.", "2 - Disagree", "4 - Agree", "4 - Agree", "2 - Disagree", 5, 6, 550, "Living room", "Family members", "Summer", "Starry night gelato", "A flowing orchestral waltz with a warm glow and gentle momentum that feels cosmic and calm"),

    # --- The Water Lily Pond ---
    ("test_021", WLP, 5, "Serene and meditative. Soft reflections create a sense of perfect stillness and inner peace.", "1 - Strongly disagree", "4 - Agree", "5 - Strongly agree", "1 - Strongly disagree", 4, 5, 600, "Living room", "Family members", "Summer", "Matcha mochi", "A soft ambient soundscape of flowing water with quiet birdsong and distant gentle wind chimes"),
    ("test_022", WLP, 4, "Deeply relaxed and at ease. Gentle colours and calm water make time feel slow and beautiful.", "1 - Strongly disagree", "5 - Strongly agree", "5 - Strongly agree", "1 - Strongly disagree", 4, 4, 400, "Living room", "Family members", "Spring", "Soft vanilla cream cake", "A slow meditative piano piece with warm harmonics and a steady gentle pulse like breathing"),
    ("test_023", WLP, 6, "Tranquil and soothing. The impressionistic style makes me feel like I am drifting on a quiet pond.", "1 - Strongly disagree", "4 - Agree", "5 - Strongly agree", "1 - Strongly disagree", 5, 5, 500, "Living room", "Family members", "Spring", "Cucumber mint sorbet", "A dreamy ambient flute melody with soft harmonic layers and a sense of gentle forward drift"),
    ("test_024", WLP, 3, "Extremely peaceful. Soft greens and pinks are restful. I could close my eyes and breathe deeply.", "1 - Strongly disagree", "5 - Strongly agree", "5 - Strongly agree", "1 - Strongly disagree", 4, 3, 350, "Bedroom", "Family members", "Summer", "Honey melon slice", "A very quiet ambient piece with barely audible tones like the hush of early morning outside"),
    ("test_025", WLP, 5, "Gentle joy. The reflected colours dance in the water. Like a peaceful Sunday morning in summer.", "1 - Strongly disagree", "4 - Agree", "5 - Strongly agree", "1 - Strongly disagree", 4, 4, 500, "Living room", "Family members", "Summer", "Fresh fruit salad", "A light airy piece with soft string harmonics and an effortless sense of warm sunlit ease"),
    ("test_026", WLP, 4, "Soft and dreamy. Like the world has slowed completely. Content just looking at the calm water.", "1 - Strongly disagree", "5 - Strongly agree", "5 - Strongly agree", "1 - Strongly disagree", 4, 4, 450, "Bedroom", "By yourself", "Spring", "Lychee panna cotta", "A slow impressionistic piano prelude with blurred harmonies and no sharp or jarring transitions"),
    ("test_027", WLP, 5, "Calming and grounding. Like a breath of fresh air. I feel fully present and completely at ease.", "1 - Strongly disagree", "4 - Agree", "5 - Strongly agree", "1 - Strongly disagree", 3, 4, 300, "Living room", "Family members", "Spring", "Green tea ice cream", "A gentle ambient soundscape with soft melodic fragments and a quiet steady background warmth"),
    ("test_028", WLP, 4, "Peaceful and nurturing. The warm greens and soft pinks are deeply comforting like a garden.", "1 - Strongly disagree", "5 - Strongly agree", "5 - Strongly agree", "1 - Strongly disagree", 4, 4, 400, "Living room", "Family members", "Summer", "Rose jam pastry", "A slow tender melody on woodwinds with a warm harmonic pad underneath and absolutely no urgency"),
    ("test_029", WLP, 6, "An invitation to quiet contemplation. A soft happiness and appreciation for beautiful stillness.", "1 - Strongly disagree", "4 - Agree", "5 - Strongly agree", "1 - Strongly disagree", 5, 5, 550, "Living room", "By yourself", "Spring", "Peach sorbet", "A meditative orchestral piece with slow string swells and a glowing sense of warmth and peace"),
    ("test_030", WLP, 3, "Soothing and beautiful. The soft brushwork makes me feel like I am floating peacefully downstream.", "1 - Strongly disagree", "5 - Strongly agree", "5 - Strongly agree", "1 - Strongly disagree", 3, 3, 250, "Bedroom", "By yourself", "Spring", "Soft coconut pudding", "A very gentle ambient hum with barely perceptible harmonic shifts and an almost silent quality"),
]

records = []
for row in rows:
    uid, painting, intensity, describe, sombre, content, calm, uneasy, colours, objects, price, room, who, season, food, soundtrack = row
    records.append({
        cols[0]:  uid,
        cols[1]:  painting,
        cols[2]:  intensity,
        cols[3]:  describe,
        cols[4]:  sombre,
        cols[5]:  content,
        cols[6]:  calm,
        cols[7]:  uneasy,
        cols[8]:  colours,
        cols[9]:  objects,
        cols[10]: price,
        cols[11]: room,
        cols[12]: who,
        cols[13]: season,
        cols[14]: food,
        cols[15]: soundtrack,
    })

out = pd.DataFrame(records)
out.to_csv("sample_test_set.csv", index=False)
print("Saved sample_test_set.csv:", out.shape)
