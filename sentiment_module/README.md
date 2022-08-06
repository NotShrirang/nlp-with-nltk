# Movie Review Analyzer

When given movie review as a string gives output whether the review is positive or negative and the confidence level of model.

### Syntax:

```lang-python
import sentiment_mod as s

print(s.sentiment("<movie review>"))
```

Example:
```lang-python
import sentiment_mod as s

print(s.sentiment("It was great movie! I liked it a lot! Lots of jokes and punchlines!"))
```
### Output:

<code> voted_classifier accuracy percent: 76.20481927710844 </code><br>
<code> ('pos', 1.0)</code>
