# AVAAS — Recording Script (template)

> **Template.** This is an example corpus using the placeholder name "Alex" (and a "SPEAKER:" dialogue marker). Replace them with your own name before recording.

## Purpose

This script is the recording corpus for fine-tuning a TTS model on Alex's voice. Total estimated read time: **~90–110 minutes** of speech across 12 sections. Variety matters more than total quantity — don't skip sections to save time.

The script is biased toward **phone-call distribution**: greetings, numbers, names, hold phrases, and conversational scenarios. Standard prose (Harvard, CMU ARCTIC) is included for phonetic and prosodic coverage.

---

## Setup

| Setting | Value |
|---|---|
| Mic | Condenser preferred (Shure MV7+, Rode NT-USB, etc.) |
| Distance | 6–8 inches, slightly off-axis |
| Sample rate | Record at 48kHz, downsample to 24kHz in post |
| Format | 16-bit mono WAV |
| Pop filter | Required |
| Room | Carpeted/soft; no AC, fan, fridge cycle, sirens |
| Levels | Peaks around -6 dBFS, no clipping |
| File naming | `{section}_{index:03d}_{slug}.wav` |

### Read style

Talk like you're actually on the phone. Not announcer-mode, not over-articulated. Pause where you'd pause. If you misread, stop, breathe, re-do that single line — don't try to fix it in post.

Take **5-minute breaks** between sections. Drink water. Splitting across 2–3 sessions on different days actually helps the final model — captures day-to-day timbre variance.

---

## Section 1: Harvard Sentences (Sets 1–10) — ~8 min

Phonetically-balanced standard corpus. Read sets 1–10 (100 sentences).

Source: https://www.cs.columbia.edu/~hgs/audio/harvard.html — save the relevant block to `data/raw/harvard_1-10.txt` and read from there.

Example (Set 1, sentences 1–3):
> The birch canoe slid on the smooth planks.
> Glue the sheet to the dark blue background.
> It's easy to tell the depth of a well.

---

## Section 2: CMU ARCTIC (first 400) — ~30 min

Continuous, single-speaker prose chosen for lexical coverage.

```bash
wget https://raw.githubusercontent.com/festvox/cmu_arctic/master/cmuarctic.data \
  -O data/raw/cmu_arctic.txt
head -n 400 data/raw/cmu_arctic.txt > data/raw/cmu_arctic_400.txt
```

Example:
> "Author of the danger trail, Philip Steels, etc."
> "Not at this particular case, Tom, apologized Whittemore."

---

## Section 3: Phone-call greetings & openers (30 lines) — ~3 min

Mix the registers — some warmer, some terser, some "answering distracted."

```
1.  Hello, this is Alex.
2.  Hi, Alex speaking.
3.  Hello?
4.  Yeah, this is Alex.
5.  Hi, you've reached Alex.
6.  Alex here.
7.  Hello, Alex speaking, how can I help?
8.  Good morning, this is Alex.
9.  Good afternoon, Alex speaking.
10. Hello — sorry, give me one second.
11. Hi there, this is Alex — go ahead.
12. Alex, who's this?
13. Hello? Hi, yeah.
14. Hi, sorry — Alex speaking.
15. Yeah hello, this is Alex.
16. Alex speaking, can you hear me okay?
17. Hi — yes, this is he.
18. Speaking.
19. Hi, Alex — what's up?
20. Hello — one sec, let me get somewhere quieter.
21. Hi, yeah, you've got me.
22. Hello, this is Alex — is everything okay?
23. Hi — Alex, go ahead.
24. Hello? Yep, this is Alex.
25. Hi there.
26. Alex, hi — how are you?
27. Hello, thanks for calling back.
28. Hi, yeah — sorry I missed your call earlier.
29. Hello? Sorry, bad signal — try me again?
30. Hi, this is Alex — what can I do for you?
```

---

## Section 4: Common phone phrases (40 lines) — ~5 min

Stock call phrases. These come up constantly.

```
1.  Can you spell that for me?
2.  Sorry, can you say that again?
3.  Let me grab a pen.
4.  One moment please.
5.  Can you hold for a second?
6.  I'm going to put you on hold briefly.
7.  Thanks for waiting.
8.  Sorry to keep you waiting.
9.  Can I take a message?
10. Can I get your number and have someone call you back?
11. What's the best number to reach you on?
12. Can you repeat the last part?
13. Sorry, you cut out for a second.
14. The line's a bit crunchy — can you say that again?
15. I'm going to need to check on that and get back to you.
16. Let me transfer you to someone who can help.
17. Bear with me a moment.
18. Could you confirm your full name for me?
19. And what's the reason for your call today?
20. Is there anything else I can help you with?
21. Thanks for calling, have a good one.
22. Talk soon.
23. Bye for now.
24. Thanks, bye.
25. Alright, take care.
26. Sounds good, talk later.
27. Okay, perfect.
28. Got it.
29. Understood.
30. That makes sense.
31. Let me look into it and circle back.
32. I'll send you an email after this call to confirm.
33. Could you email me that as well, just so I have a record?
34. Yes, that works for me.
35. No, that won't work, unfortunately.
36. Can we move it to a different time?
37. I'm sorry, I'm not able to help with that.
38. That's a question for a different team — let me find you the right person.
39. I'll need to escalate this — what's the best way to reach you tomorrow?
40. Apologies for the confusion.
```

---

## Section 5: Numbers in context (60 lines) — ~7 min

Numbers are where TTS models break most often. Read each as you'd actually say it — natural groupings, not robotic digit-by-digit unless the line specifies.

### 5a. Phone numbers (15)

```
1.  My number is four one five, five five five, oh one three two.
2.  You can reach me at two oh two, five five five, eight nine one one.
3.  The office line is six four six, five five five, twelve hundred.
4.  Try eight hundred, five five five, ninety-eight seventy-six.
5.  Mobile is plus four four, seven seven hundred, nine hundred thousand.
6.  Extension is two two seven.
7.  Press one for English, two para español.
8.  Dial nine to get an outside line.
9.  My fax — yes, fax — is three oh five, five five five, four four four four.
10. Country code plus eight one, then three, then five five five five, six six six six.
11. The toll-free is one eight six six, five five five, zero one two three.
12. I'll text it to you: four one five, two zero two, ninety-eight oh four.
13. The conference bridge is eight five five, five five five, two two two two, code nine eight seven six five four.
14. International code is plus one for US and Canada.
15. The shortcode is four seven six four seven.
```

### 5b. Addresses (15)

```
1.  Forty-seven thirty-two Mission Street.
2.  Twelve fifty Sixth Avenue, apartment four B.
3.  Suite three hundred.
4.  The address is one hundred Main Street, Cambridge, Mass.
5.  Twenty-eight forty Pine, unit six.
6.  Number seven, Willow Lane.
7.  The zip is nine four one one zero.
8.  Postcode is W one A, one A A.
9.  The corner of Twenty-third and Valencia.
10. Building eight, floor twelve.
11. PO Box one four three two, San Francisco.
12. Sixteen Elm Court, that's E L M.
13. The unit number is seventeen oh six.
14. Cross street is Folsom.
15. It's the third house on the left after the stop sign.
```

### 5c. Dates and times (15)

```
1.  Thursday, March twelfth.
2.  April second, twenty twenty-six.
3.  The seventeenth at three pm.
4.  Quarter past four.
5.  Half past seven.
6.  Nine thirty in the morning.
7.  Twenty-two hundred hours.
8.  Sometime after lunch, maybe two-ish.
9.  End of business tomorrow.
10. By close of play Friday.
11. The first of next month.
12. A week from Tuesday.
13. Christmas Eve.
14. New Year's Day.
15. The third Wednesday of every month.
```

### 5d. Money and codes (15)

```
1.  Twelve dollars and forty-seven cents.
2.  A thousand bucks even.
3.  Two point five million.
4.  Three hundred grand.
5.  Forty-nine ninety-nine.
6.  Twenty quid.
7.  My confirmation code is alpha four seven gamma nine.
8.  Order number is W X seven seven two one zero.
9.  Reference: bravo bravo dash eight oh four.
10. Tracking number one Z, nine nine nine, A A A, ten, one two three, four five six, seven eight nine zero.
11. License plate eight K Z U, six six four.
12. The PIN is two four six eight.
13. Last four of my social are four four three two.
14. My account ends in nine eight seven six.
15. Verification code seven nine one three four two.
```

---

## Section 6: Names & spelling (40 lines) — ~5 min

Names are the second-biggest failure mode after numbers.

### 6a. Common first names — say each naturally (20)

```
James, Mary, John, Patricia, Robert, Jennifer, Michael, Linda,
William, Elizabeth, David, Barbara, Richard, Susan, Joseph,
Jessica, Thomas, Sarah, Christopher, Karen.
```

### 6b. Common last names (10)

```
Smith, Johnson, Williams, Brown, Jones, Garcia, Miller, Davis,
Rodriguez, Martinez.
```

### 6c. Spelling out (10)

Read each name, then spell it letter-by-letter at natural pace:

```
1.  Alex — A, L, E, X.
2.  Anjali — A, N, J, A, L, I.
3.  Iyer — I, Y, E, R.
4.  Chen — C, H, E, N.
5.  O'Brien — O, apostrophe, B, R, I, E, N.
6.  Schmidt — S, C, H, M, I, D, T.
7.  Nguyen — N, G, U, Y, E, N.
8.  Müller — M, U with umlaut, L, L, E, R.
9.  Caoimhín — C, A, O, I, M, H, I with fada, N.
10. Søren — S, O with slash, R, E, N.
```

---

## Section 7: Alphabet drills (4 takes) — ~2 min

```
Take 1 (NATO phonetic, normal pace):
  Alpha, Bravo, Charlie, Delta, Echo, Foxtrot, Golf, Hotel,
  India, Juliet, Kilo, Lima, Mike, November, Oscar, Papa,
  Quebec, Romeo, Sierra, Tango, Uniform, Victor, Whiskey,
  X-ray, Yankee, Zulu.

Take 2 (Plain letters, slow + deliberate):
  A. B. C. D. E. F. G. H. I. J. K. L. M.
  N. O. P. Q. R. S. T. U. V. W. X. Y. Z.

Take 3 (Plain letters, conversational pace):
  A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P,
  Q, R, S, T, U, V, W, X, Y, Z.

Take 4 (Digits 0–9, both ways):
  Zero, one, two, three, four, five, six, seven, eight, nine.
  Oh, one, two, three, four, five, six, seven, eight, nine.
```

---

## Section 8: Questions & confirmations (60 lines) — ~6 min

### 8a. Questions — rising intonation (20)

```
1.  Did you get my email?
2.  Is now a good time?
3.  Can you hear me okay?
4.  Are you still there?
5.  Does that work for you?
6.  Is that the right number?
7.  Did I catch you at a bad time?
8.  Could we move this to Monday?
9.  Should I send it to your work email or personal?
10. Have you spoken to her about it yet?
11. Is everyone on the call?
12. Did the package arrive?
13. Are we good?
14. Can I follow up later this week?
15. Would you mind sending that over in writing?
16. Did anything else come up?
17. Is there a deadline I should know about?
18. Can we loop in someone from finance?
19. Was that figure before or after tax?
20. Did you want me to handle it from here?
```

### 8b. Tag questions (10)

```
1.  That works, doesn't it?
2.  You're available Thursday, right?
3.  They confirmed already, didn't they?
4.  You got the file, didn't you?
5.  It's the same as last time, isn't it?
6.  We agreed on the third, right?
7.  You're not coming in tomorrow, are you?
8.  She mentioned that, didn't she?
9.  It's billed monthly, not annually, right?
10. You don't need me on this one, do you?
```

### 8c. Confirmations & acknowledgments (15)

```
1.  Yes, that's right.
2.  Yep, exactly.
3.  Correct.
4.  That's the one.
5.  Got it, thanks.
6.  Mm-hmm, makes sense.
7.  Sure thing.
8.  No problem.
9.  Absolutely.
10. Of course.
11. Sounds good to me.
12. Yeah, I'm on board.
13. I'm with you.
14. Roger that.
15. Affirmative.
```

### 8d. Polite refusals & corrections (15)

```
1.  I'm afraid that won't work for me.
2.  Sorry, that's not quite right — let me clarify.
3.  Actually, I think you mean the other one.
4.  No, that's not what I was suggesting.
5.  I appreciate the offer, but I'll have to pass.
6.  Hmm, I don't think so — let me double-check.
7.  That's not how I remember it.
8.  I'd rather not, if it's all the same.
9.  Sorry, I won't be able to make that.
10. Unfortunately I'm booked.
11. Let me push back gently — I don't think that's the right approach.
12. I disagree, but I see where you're coming from.
13. That's a stretch, honestly.
14. I'm not sure I follow.
15. Could you explain that one more time?
```

---

## Section 9: Conversational scenarios (5 × ~10 turns) — ~12 min

Read **only Alex's lines** (marked SPEAKER). Pause naturally where the other person would respond — those gaps give the model realistic turn-taking timing.

### 9a. Restaurant reservation

```
SPEAKER: Hi, I'd like to make a reservation for Friday night.
SPEAKER: Yeah, for two people.
SPEAKER: Around seven thirty if you have it.
SPEAKER: That works. Under the name Alex — A, L, E, X.
SPEAKER: Yep, four one five, five five five, ninety-eight oh two.
SPEAKER: Just a high-chair, actually, if you have one.
SPEAKER: Perfect, thank you.
SPEAKER: We'll see you then.
```

### 9b. Tech support call (as caller)

```
SPEAKER: Hi — I'm having trouble with my internet, it keeps dropping out.
SPEAKER: It's been happening since yesterday evening.
SPEAKER: Yeah, I've tried unplugging the router for thirty seconds and plugging it back in.
SPEAKER: No change.
SPEAKER: The lights on the front are all green except the WAN light, which is flashing orange.
SPEAKER: Account number is one one four eight, two two three six.
SPEAKER: How long is the outage expected to last?
SPEAKER: Okay, I'll wait it out then. Will you send a text when it's restored?
SPEAKER: Thanks, appreciate it.
```

### 9c. Scheduling a meeting

```
SPEAKER: Hey, I wanted to find a time for us to sync about the proposal.
SPEAKER: Probably half an hour should do it.
SPEAKER: I'm pretty open Wednesday afternoon — say, two pm?
SPEAKER: If that doesn't work, Thursday morning would be good too.
SPEAKER: Let's do Wednesday at two then.
SPEAKER: I'll send a calendar invite with the dial-in.
SPEAKER: Should I include Priya, or do you want to brief her separately?
SPEAKER: Got it, I'll keep it just the two of us.
SPEAKER: Talk Wednesday.
```

### 9d. Bank verification call (as customer)

```
SPEAKER: Hi, I got a text asking me to call about a transaction.
SPEAKER: My account ends in nine eight seven six.
SPEAKER: Date of birth is January twenty-second, nineteen eighty-eight.
SPEAKER: Last four of my social — four four three two.
SPEAKER: Yes, I made that purchase. It was me.
SPEAKER: Forty-two dollars, that sounds right.
SPEAKER: No other unfamiliar charges that I can see, no.
SPEAKER: Great, thanks for confirming.
SPEAKER: Should I do anything else, or are we all set?
SPEAKER: Perfect, thanks for your help.
```

### 9e. Returning a missed call

```
SPEAKER: Hi, this is Alex — I'm returning a call from earlier today.
SPEAKER: Yeah, I missed it around eleven.
SPEAKER: The voicemail said something about a delivery?
SPEAKER: Oh I see — and they need someone to be present to sign?
SPEAKER: Tomorrow afternoon works.
SPEAKER: Between two and five, ideally.
SPEAKER: Yeah, that's the right address. Forty-seven thirty-two Mission Street, apartment six.
SPEAKER: Great, thank you. I'll be there.
SPEAKER: Bye now.
```

---

## Section 10: Spontaneous prompts — ~15 min

For each prompt, talk freely for 60–90 seconds. **Don't read** — actually answer. This captures unscripted prosody, fillers ("um", "uh", "y'know"), and how you sound when thinking rather than reading.

```
1.  Describe what you did this past weekend.
2.  Talk through what you had for breakfast and why.
3.  Explain what a podcast is to a friend who doesn't know.
4.  Describe your daily routine.
5.  Talk about a movie or show you watched recently.
6.  Explain how you got into hardware hacking.
7.  Describe your workspace — what's on the desk right now.
8.  Talk about a problem you're trying to solve at work.
9.  Describe a memorable trip you took.
10. Explain how an LLM works to a curious non-technical friend.
11. Talk about something that frustrated you recently.
12. Describe your favorite meal in detail.
13. Talk about what you're looking forward to this year.
14. Explain why you chose this voice-cloning approach (meta).
15. Just freestyle for 90 seconds about whatever's on your mind.
```

---

## Section 11: Voicemail greetings & sign-offs (10) — ~2 min

```
1.  Hi, you've reached Alex. I can't take your call right now — leave a message and I'll get back to you. Thanks.
2.  Hey, it's Alex. Leave a message.
3.  Alex here — sorry I missed you, try me again or leave a message.
4.  You've reached Alex's voicemail. Please leave your name, number, and a brief message.
5.  Hi, it's Alex — I'm away until Monday. For urgent matters please contact my assistant.
6.  Thanks for calling, leave a message at the beep.
7.  Hi, this is Alex. I'll call you right back.
8.  Alright, sounds good — talk to you later.
9.  Have a great rest of your day.
10. Thanks again, bye.
```

---

## Section 12: Edge cases & emotional register (20) — ~3 min

False starts, restarts, and varied emotional registers. Natural speech is full of these.

```
1.  Wait — sorry, scratch that.
2.  Let me start over.
3.  Actually no, the other one.
4.  Hold on, I had it backwards.
5.  Sorry — let me think for a second.
6.  Hmm.
7.  Yeah... yeah, okay.
8.  Uh-huh.
9.  Mmm.
10. Oh! Right, of course.
11. Hang on, my phone's dying.
12. Sorry, my dog's going crazy in the background.
13. I missed that — could you repeat the last part?
14. You're breaking up.
15. Lost you for a second there.
16. (laughing) Oh man, yeah.
17. (sighing) Okay, okay.
18. (whispering) Sorry, I'm in a meeting — can I call you right back?
19. (frustrated) No, that's not what I said.
20. (warm) Thanks, that really means a lot.
```

---

## After recording

1. Run `scripts/preprocess.py` to:
   - Downsample 48kHz → 24kHz
   - Normalize loudness to -23 LUFS
   - Trim silence > 1.5s (keep short pauses — they're training signal)
   - Auto-transcribe with Whisper-large-v3 for alignment
   - Output a training-ready manifest at `data/processed/manifest.jsonl`

2. **Manual QC**: spot-check ~10% of segments. Reject anything with clicks, breaths-too-loud, sirens, or misreads.

3. Once you have **at least 60 minutes of clean post-QC audio**, kick off training (see `training/`).

---

## Quality checklist before training

- [ ] All sections recorded across at least 2 sessions
- [ ] Total clean (post-QC) duration ≥ 60 min, ideally 90+
- [ ] No section under-represented (each ≥80% of its target length)
- [ ] Whisper transcription WER < 3% (proxy for clarity)
- [ ] Loudness within ±2 LU of -23 LUFS across all clips
- [ ] No clipping (peak < -1 dBFS on any clip)
