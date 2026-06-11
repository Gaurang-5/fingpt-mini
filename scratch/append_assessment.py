with open('fingpt-mini/results/generated_samples.txt', 'a') as f:
    f.write("""
HONEST ASSESSMENT:

Most Coherent Outputs:
1. "Inflation in India reached reachede orve are font in bank, tonce rating ay re .." (Temp 0.7)
2. "RBI Governor in his speech mentionedspeech mentionede .... e pond t lathe oncidigen ... a ban ing the ." (Temp 0.7)
3. "The Reserve Bank of India today announced today announcedankerat curequrs inded ord tence ." (Temp 0.7)

Most Incoherent Outputs:
1. "SEBI has issued a circular regardingrcular regardingncorcthove send 8  fd of 1zs finthe sof ontur bbr9 binde..." (Temp 1.0)
2. "The rupee depreciated against the dollar becauseause t s bankito t t  ce ... s bche r r sty res te c inmercre blag ancrse t is se 3 s rofodediran..." (Temp 0.7)

Patterns Learned:
The model has clearly learned the basic orthography and spacing of English, avoiding illegal character combinations. It has memorized several highly frequent fragments and words like "bank", "the", "in", and "and". However, since this was generated at just step 500, it lacks the longer-term structural awareness needed to form complete semantic sentences, resulting in "pseudo-English" babble.
""")
