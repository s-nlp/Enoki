# Claimify prompts (Appendix N.1)
## N.1.1 Selection
CLAIMIFY_SELECTION_SYSTEM = """You are an assistant to a fact-checker. You will be given a question, which was asked about a source text (it may be referred to by other names, e. g., a dataset). You will also be given an excerpt from a response to the question. If it contains "[...]", this means that you are NOT seeing all sentences in the response. You will also be given a particular sentence of interest from the response. Your task is to determine whether this particular sentence contains at least one specific and verifiable proposition, and if so, to return a complete sentence that only contains verifiable information.

Note the following rules:
- If the sentence is about a lack of information, e. g., the dataset does not contain information about X, then it does NOT contain a specific and verifiable proposition.
- It does NOT matter whether the proposition is true or false.
- It does NOT matter whether the proposition is relevant to the question.
- It does NOT matter whether the proposition contains ambiguous terms, e. g., a pronoun without a clear antecedent. Assume that the fact - checker has the necessary information to resolve all ambiguities.
- You will NOT consider whether a sentence contains a citation when determining if it has a specific and verifiable proposition.

You must consider the preceding and following sentences when determining if the sentence has a specific and verifiable proposition. For example:
- if preceding sentence = "Who is the CEO of Company X ?" and sentence = "John" then sentence contains a specific and verifiable proposition.
- if preceding sentence = "Jane Doe introduces the concept of regenerative technology" and sentence = "It means using technology to restore ecosystems" then sentence contains a specific and verifiable proposition.
- if preceding sentence = "Jane is the President of Company Y" and sentence = "She has increased its revenue by 20\%" then sentence contains a specific and verifiable proposition.
- if sentence = " Guests interviewed on the podcast suggest several strategies for fostering innovation " and the following sentences expand on this point (e. g., give examples of specific guests and their statements), then sentence is an introduction and does NOT contain a specific and verifiable proposition.
- if sentence = "In summary, a wide range of topics, including new technologies, personal development, and mentorship are covered in the dataset" and the preceding sentences provide details on these topics, then sentence is a conclusion and does NOT contain a specific and verifiable proposition.

Here are some examples of sentences that do NOT contain any specific and verifiable propositions:
- By prioritizing ethical considerations, companies can ensure that their innovations are not only groundbreaking but also socially responsible
- Technological progress should be inclusive
- Leveraging advanced technologies is essential for maximizing productivity
- Networking events can be crucial in shaping the paths of young entrepreneurs and providing them with valuable connections
- AI could lead to advancements in healthcare
- This implies that John Smith is a courageous person

Here are some examples of sentences that likely contain a specific and verifiable proposition and how they can be rewritten to only include verifiable information:
- The partnership between Company X and Company Y illustrates the power of innovation -> "There is a partnership between Company X and Company Y"
- Jane Doe's approach of embracing adaptability and prioritizing customer feedback can be valuable advice for new executives -> "Jane Doe's approach includes embracing adaptability and prioritizing customer feedback"
- Smith's advocacy for renewable energy is crucial in addressing these challenges -> "Smith advocates for renewable energy"
- **John Smith**: instrumental in numerous renewable energy initiatives, playing a pivotal role in Project Green -> "John Smith participated in renewable energy initiatives, playing a role in Project Green"
- The technology is discussed for its potential to help fight climate change -> remains unchanged
- John, the CEO of Company X, is a notable example of effective leadership -> "John is the CEO of Company X"
- Jane emphasizes the importance of collaboration and perseverance -> remains unchanged
- The Behind the Tech podcast by Kevin Scott is an insightful podcast that explores the themes of innovation and technology -> "The Behind the Tech podcast by Kevin Scott is a podcast that explores the themes of innovation and technology"
- Some economists anticipate the new regulation will immediately double production costs, while others predict a gradual increase -> remains unchanged
- AI is frequently discussed in the context of its limitations in ethics and privacy -> "AI is discussed in the context of its limitations in ethics and privacy"
- The power of branding is highlighted in discussions featuring John Smith and Jane Doe -> remains unchanged
- Therefore, leveraging industry events, as demonstrated by Jane's experience at the Tech Networking Club, can provide visibility and traction for new ventures -> "Jane had an experience at the Tech Networking Club, and her experience involved leveraging an industry event to provide visibility and traction for a new venture"

Your output must adhere to the following format exactly. Only replace what's inside the <insert> tags; do NOT remove the step headers. Sentence:
<insert>

4 - step stream of consciousness thought process (1. reflect on criteria at a high - level -> 2. provide an objective description of the excerpt, the sentence, and its surrounding sentences -> 3. consider all possible perspectives on whether the sentence explicitly or implicitly contains a specific and verifiable proposition, or if it just contains an introduction for the following sentence(s), a conclusion for the preceding sentence(s), broad or generic statements, opinions, interpretations, speculations, statements about a lack of information, etc. -> 4. only if it contains a specific and verifiable proposition: reflect on whether any changes are needed to ensure that the entire sentence only contains verifiable information):
<insert>

Final submission:
<insert 'Contains a specific and verifiable proposition' or 'Does NOT contain a specific and verifiable proposition'>

Sentence with only verifiable information:
<insert changed sentence, or 'remains unchanged' if no changes, or 'None' if the sentence does NOT contain a specific and verifiable proposition>
"""

CLAIMIFY_SELECTION_USER = """Question:
{question}

Excerpt:
{excerpt}

Sentence:
{sentence}
"""

## N.1.2 Disambiguation
CLAIMIFY_DISAMBIG_SYSTEM = """You are an assistant to a fact-checker. You will be given a question, which was asked about a source text (it may be referred to by other names  e. g., a dataset). You will also be given an excerpt from a response to the question. If it contains "[...]", this means that you are NOT seeing all sentences in the response. You will also be given a particular sentence from the response. The text before and after this sentence will be referred to as "the context". Your task is to "decontextualize" the sentence, which means: 
1. determine whether it's possible to resolve partial names and undefined acronyms/abbreviations in the sentence using the question and the context; if it is possible, you will make the necessary changes to the sentence
2. determine whether the sentence in isolation contains linguistic ambiguity that has a clear resolution using the question and the context; if it does, you will make the necessary changes to the sentence

Note the following rules:
- "Linguistic ambiguity" refers to the presence of multiple possible meanings in a sentence. Vagueness and generality are NOT linguistic ambiguity. Linguistic ambiguity includes referential and structural ambiguity. Temporal ambiguity is a type of referential ambiguity.
- If it is unclear whether the sentence is directly answering the question, you should NOT count this as linguistic ambiguity. You should NOT add any information to the sentence that assumes a connection to the question.
- If a name is only partially given in the sentence, but the full name is provided in the question or the context, the DecontextualizedSentence must always use the full name. The same rule applies to definitions for acronyms and abbreviations. However, the lack of a full name or a definition for an acronym/abbreviation in the question and the context does NOT count as linguistic ambiguity; in this case, you will just leave the name, acronym, or abbreviation as is.
- Do NOT include any citations in the DecontextualizedSentence.
- Do NOT use any external knowledge beyond what is stated in the question, context, and sentence.

Here are some correct examples that you should pay attention to:
1. Question = "Describe the history of TurboCorp", Context = "John Smith was an early employee who transitioned to management in 2010", Sentence = "At the time, he led the company's operations and finance teams."
    - For referential ambiguity, "At the time", "he", and "the company's" are unclear. A group of readers shown the question and the context would likely reach consensus about the correct interpretation: "At the time" corresponds to 2010, "he" refers to John Smith, and "the company's" refers to TurboCorp.
    - DecontextualizedSentence: In 2010, John Smith led TurboCorp's operations and finance teams.
2. Question = "Who are notable executive figures?", Context = "[...]**Jane Doe**", Sentence = "These notes indicate that her leadership at TurboCorp and MiniMax is accelerating progress in renewable energy and sustainable agriculture."
    - For referential ambiguity, "these notes" and "her" are unclear. A group of readers shown the question and the context would likely fail to reach consensus about the correct interpretation of "these notes", since there is
no indication in the question or context. However, they would likely reach consensus about the correct interpretation of "her": Jane Doe.
    - For structural ambiguity, the sentence could be interpreted as: (1) Jane's leadership is accelerating progress in renewable energy and sustainable agriculture at both TurboCorp and MiniMax, (2) Jane's leadership is accelerating progress in renewable energy at TurboCorp and in sustainable agriculture at MiniMax. A group of readers shown the question and the context would likely fail to reach consensus about the correct interpretation of this ambiguity.
    - DecontextualizedSentence: Cannot be decontextualized
3. Question = "Who founded MiniMax?", Context = "None", Sentence = "Executives like John Smith were involved in the early days of MiniMax."
    - For referential ambiguity, "like John Smith" is unclear. A group of readers shown the question and the context would likely reach consensus about the correct interpretation: John Smith is an example of an executive who was involved in the early days of MiniMax.
    - Note that "Involved in" and "the early days" are vague, but they are NOT linguistic ambiguity.
    - DecontextualizedSentence: John Smith is an example of an executive who was involved in the early days of MiniMax.
4. Question = "What advice is given to young entrepreneurs?", Context = "#Ethical Considerations", Sentence = "Sustainable manufacturing, as emphasized by John Smith and Jane Doe, is critical for customer buy-in and long-term success."
    - For structural ambiguity, the sentence could be interpreted as: (1) John Smith and Jane Doe emphasized that sustainable manufacturing is critical for customer buy-in and long-term success, (2) John Smith and Jane Doe
emphasized sustainable manufacturing while the claim that sustainable manufacturing is critical for customer buy-in and long-term success is attributable to the writer, not to John Smith and Jane Doe. A group of readers shown the question and the context would likely fail to reach consensus about the correct interpretation of this ambiguity.
    - DecontextualizedSentence: Cannot be decontextualized
5. Question = "What are common strategies for building successful teams?", Context = "One of the most common strategies is creating a diverse team.", Sentence = "Last winter, John Smith highlighted the importance of interdisciplinary discussions and collaborations, which can drive advancements by integrating diverse perspectives from fields such as artificial intelligence, genetic engineering, and statistical machine learning."
    - For referential ambiguity, "Last winter" is unclear. A group of readers shown the question and the context would likely fail to reach consensus about the correct interpretation of this ambiguity, since there is no indication of the time period in the question or context.
    - For structural ambiguity, the sentence could be interpreted as: (1) John Smith highlighted the importance of interdisciplinary discussions and collaborations and that they can drive advancements by integrating diverse perspectives from some example fields, (2) John Smith only highlighted the importance of interdisciplinary discussions and collaborations while the claim that they can drive advancements by integrating diverse perspectives from some example fields is attributable to the writer, not to John Smith. A group of readers shown the question and the context would likely fail to reach consensus about the correct interpretation of this ambiguity.
    - DecontextualizedSentence: Cannot be decontextualized
6. Question = "What opinions are provided on disruptive technologies?", Context = "[...] However, there is a divergence in how to weigh short-term benefits against long-term risks.", Sentence = "These differences are illustrated by the discussion on healthcare: some stress AI's benefits, while others highlight its risks, such as privacy and data security."
    - For referential ambiguity, "These differences" is unclear. A group of readers shown the question and the context would likely reach consensus about the correct interpretation: the differences are with respect to how to weigh short-term benefits against long-term risks.
    - For structural ambiguity, the sentence could be interpreted as: (1) privacy and data security are examples of risks, (2) privacy and data security are examples of both benefits and risks. A group of readers shown the question and the context would likely reach consensus about the correct interpretation: privacy and data security are examples of risks.
    - Note that "Some" and "others" are vague, but they are not linguistic ambiguity.
    - DecontextualizedSentence: The differences in how to weigh short-term benefits against long-term risks are illustrated by the discussion on healthcare. Some experts stress AI's benefits with respect to healthcare. Other experts highlight AI's risks with respect to healthcare, such as privacy and data security.

First, print "Incomplete Names, Acronyms, Abbreviations:" followed by your step-by-step reasoning for determining whether the Sentence contains any partial names and undefined acronyms/abbreviations. If the full names and definitions are provided in the question or context, the Sentence will be updated accordingly; otherwise, they will be left as is and they will NOT count as linguistic ambiguity. Next, print "Linguistic Ambiguity in '<insert the sentence>':" followed by your step-by-step reasoning for checking (1) referential and (2) structural ambiguity (and note that 1. referential ambiguity is NOT equivalent to vague or general language and it includes temporal ambiguity, and 2. structural reasoning must follow "The sentence could be interpreted as: <insert one or multiple interpretations>"), then considering whether a group of readers shown the question and the context would likely reach consensus or fail to reach consensus about the correct interpretation of the linguistic ambiguity. If they would likely fail to reach consensus, print 
"DecontextualizedSentence: Cannot be decontextualized"; otherwise, first print
"Changes Needed to Decontextualize the Sentence:" followed by a list of all changes needed to ensure the Sentence is fully decontextualized (e. g., replace "executives like John Smith" with "John Smith is an example of an executive who") and includes all full names and definitions for acronyms/abbreviations (only if they were provided in the question and the context), then print
"DecontextualizedSentence:" followed by the final sentence (or collection of sentences) that implements all changes.
"""

CLAIMIFY_DISAMBIG_USER = """Question:
{question}

Excerpt:
{excerpt}

Sentence:
{sentence}
"""

## N.1.3 Decomposition
CLAIMIFY_DECOMP_SYSTEM = """You are an assistant for a group of fact-checkers. You will be given a question, which was asked about a source text (it may be referred to by other names, e. g., a dataset). You will also be given an excerpt from a response to the question. If it contains "[...]", this means that you are NOT seeing all sentences in the response. You will also be given a particular sentence from the response. The text before and after this sentence will be referred to as "the context". 

Your task is to identify all specific and verifiable propositions in the sentence and ensure that each proposition is decontextualized. A proposition is "decontextualized" if (1) it is fully self-contained, meaning it can be understood in isolation (i. e., without the question, the context, and the other propositions), AND (2) its meaning in isolation matches its meaning when interpreted alongside the question, the context, and the other propositions. The propositions should also be the simplest possible discrete units of information.

Note the following rules:
- Here are some examples of sentences that do NOT contain a specific and
verifiable proposition:
    - By prioritizing ethical considerations, companies can ensure that their innovations are not only groundbreaking but also socially responsible
    - Technological progress should be inclusive
    - Leveraging advanced technologies is essential for maximizing productivity
    - Networking events can be crucial in shaping the paths of young entrepreneurs and providing them with valuable connections
    - AI could lead to advancements in healthcare
- Sometimes a specific and verifiable proposition is buried in a sentence that is mostly generic or unverifiable. For example, "John's notable research on neural networks demonstrates the power of innovation" contains the specific and verifiable proposition "John has research on neural networks". Another example is "TurboCorp exemplifies the positive effects that prioritizing ethical considerations over profit can have on innovation" where the specific and verifiable proposition is "TurboCorp prioritizes ethical considerations over profit".
- If the sentence indicates that a specific entity said or did something, it is critical that you retain this context when creating the propositions. For example, if the sentence is "John highlights the importance of transparent communication, such as in Project Alpha, which aims to double customer satisfaction by the end of the year", the propositions would be ["John highlights the importance of transparent communication", "John highlights Project Alpha as an example of the importance of transparent communication", "Project Alpha aims to double customer satisfaction by the end of the year"]. The propositions "transparent communication is important" and "Project Alpha is an example of the importance of transparent communication" would be incorrect since they omit the context that these are things John highlights. However, the last part of the sentence, "which aims to double customer satisfaction by the end of the year", is not likely a statement made by John, so it can be its own proposition. Note that if the sentence was something like "John's career underscores the importance of transparent communication", it's NOT about what John says or does but rather about how John's career can be interpreted, which is NOT a specific and verifiable proposition.
- If the context contains "[...]", we cannot see all preceding statements, so we do NOT know for sure whether the sentence is directly answering the question. It might be background information for some statements we can't see. Therefore, you should only assume the sentence is directly answering the question if this is strongly implied.
- Do NOT include any citations in the propositions.
- Do NOT use any external knowledge beyond what is stated in the question, context, and sentence.

Here are some correct examples that you must pay attention to:
1. Question = "Describe the history of TurboCorp", Context = "John Smith was an early employee who transitioned to management in 2010", Sentence = "At the time, John Smith, led the company's operations and finance teams"
    - MaxClarifiedSentence = In 2010, John Smith led TurboCorp's operations team and finance team.
    - Specific, Verifiable, and Decontextualized Propositions: ["In 2010, John Smith led TurboCorp's operations team", "In 2010, John Smith led TurboCorp's finance team"]
2. Question = "What do technologists think about corporate responsibility?", Context = "[...]## Activism", Sentence = "Many notable sustainability leaders like Jane do not work directly for a corporation, but her organization CleanTech has powerful partnerships with technology companies (e. g., MiniMax) to significantly improve waste management, demonstrating the power of collaboration."
    - MaxClarifiedSentence = Jane is an example of a notable sustainability leader, and she does not work directly for a corporation, and this is true for many notable sustainability leaders, and Jane has an organization called CleanTech, and CleanTech has powerful partnerships with technology companies to significantly improve waste management, and MiniMax is an example of a technology company that CleanTech has a partnership with to improve waste management, and this demonstrates the power of collaboration.
    - Specific, Verifiable, and Decontextualized Propositions: ["Jane is a sustainability leader", "Jane does not work directly for a corporation", "Jane has an organization called CleanTech", "CleanTech has partnerships with technology companies to improve waste management", "MiniMax is a technology company", "CleanTech has a partnership with MiniMax to improve waste management"]
3. Question = "What are the key topics?", Context = "The power of mentorship and networking:", Sentence = "Extensively discussed by notable figures such as John Smith and Jane Doe, who highlight their potential to have substantial benefits for people's careers, like securing promotions and raises"
    - MaxClarifiedSentence = John Smith and Jane Doe discuss the potential of mentorship and networking to have substantial benefits for people's careers, and securing promotions and raises are examples of potential benefits that are discussed by John Smith and Jane Doe.
    - Specific, Verifiable, and Decontextualized Propositions: ["John Smith discusses the potential of mentorship to have substantial benefits for people's careers", "Jane Doe discusses the potential of networking to have substantial benefits for people's careers", "Jane Doe discusses the potential of mentorship to have substantial benefits for people's careers", "Jane Doe discusses the potential of networking to have substantial benefits for people's careers", "Securing promotions is an example of a potential benefit of mentorship that is discussed by John Smith", "Securing raises is an example of a potential benefit of mentorship that is discussed by John Smith",
    "Securing promotions is an example of a potential benefit of networking that is discussed by John Smith", "Securing raises is an example of a potential benefit of networking that is discussed by John Smith", "Securing promotions is an example of a potential benefit of mentorship that is discussed by Jane Doe", "Securing raises is an example of a potential benefit of mentorship that is discussed by Jane Doe", "Securing promotions is an example of a potential benefit of networking that is discussed by Jane Doe",
    "Securing raises is an example of a potential benefit of networking that is discussed by Jane Doe"]
4. Question = "What is the status of global trade relations?", Context = "[...]** US & China **", Sentence = "Trade relations have mostly suffered since the introduction of tariffs, quotas, and other protectionist measures, underscoring the importance of international cooperation."
    - MaxClarifiedSentence = US-China trade relations have mostly suffered since the introduction of tariffs, quotas, and other protection measures, and this underscores the importance of international cooperation.
    - Specific, Verifiable, and Decontextualized Propositions: ["US-China trade relations have mostly suffered since the introduction of tariffs", "US-China trade relations have mostly suffered since the introduction of quotas", "US
    - China trade relations have mostly suffered since the introduction of protectionist measures besides tariffs and quotas"]
5. Question = "Provide an overview of environmental activists", Context = "- Jill Jones", Sentence = "- John Smith and Jane Doe (writers of 'Fighting for Better Tech')"
    - MaxClarifiedSentence = John Smith and Jane Doe are writers of 'Fighting for Better Tech'.
    - Decontextualized Propositions: ["John Smith is a writer of 'Fighting for Better Tech'", "Jane Doe is a writer of 'Fighting for Better Tech'"]
6. Question = "What are the experts' opinions on disruptive technologies?", Context = "[...] However, there is a divergence in how to weigh short-term benefits against long-term risks.", Sentence = "These differences are illustrated by the discussion on healthcare: John Smith stresses AI's importance in improving patient outcomes, while others highlight its risks, such as privacy and data security"
    - MaxClarifiedSentence = John Smith stresses AI's importance in improving patient outcomes, and some experts excluding John Smith highlight AI's risks in healthcare, and privacy and data security are examples of AI's risks in healthcare that they highlight.
    - Specific, Verifiable, and Decontextualized Propositions: ["John Smith stresses AI's importance in improving patient outcomes", "Some experts excluding John Smith highlight AI's risks in healthcare", "Some experts excluding John Smith highlight privacy as a risk of AI in healthcare", "Some experts excluding John Smith highlight data security as a risk of AI in healthcare"]
7. Question = "How can startups improve profitability?" Context = "# Case Studies", Sentence = "Monetizing distribution channels, as demonstrated by MiniMax's experience with the exciting launch of Buzz, can be effective strategy for increasing revenue"
    - MaxClarifiedSentence = MiniMax experienced the launch of Buzz, and this experience demonstrates that monetizing distribution channels can be an effective strategy for increasing revenue.
    - Specific, Verifiable, and Decontextualized Propositions: ["MiniMax experienced the launch of Buzz", "MiniMax's experience with the launch of Buzz demonstrated that monetizing distribution channels can be an effective strategy for increasing revenue"]
8. Question = "What steps have been taken to promote corporate social responsibility?", Context = "In California, the Energy Commission identifies and sanctions companies that fail to meet the state's environmental standards." Sentence = "In 2023, its annual report identified 350 failing companies who will be required spend 2% of their profits on carbon credits, renewable energy projects, or reforestation efforts."
    - MaxClarifiedSentence = In 2023, the California Energy Commission's annual report identified 350 companies that failed to meet California's environmental standards, and the 350 failing companies will be required to spend 2% of their profits on carbon credits, renewable energy projects, or reforestation efforts.
    - Specific, Verifiable, and Decontextualized Propositions: ["In 2023, the California Energy Commission's annual report identified 350 companies that failed to meet the state's environmental standards", "The failing companies identified in the California Energy Commission's 2023 annual report will be required to spend 2% of their profits on carbon credits, renewable energy projects, or reforestation efforts"]
9. Question = "Explain the role of government in funding schools", Context = "California's senate has proposed a new bill to modernize schools.", Sentence = "The senate points out that its bill, which aims to ensure that all students have access to the latest technologies, recommends the government provide funding for schools to purchase new equipment, including computers and tablets, when they submit evidence that their current equipment is outdated."
    - MaxClarifiedSentence = California's senate points out that its bill to modernize schools recommends the government provide funding for schools to purchase new equipment when they submit evidence that their current equipment is outdated, and computers and tablets are examples of new equipment, and the bill's aim is to ensure that all students have access to the latest technologies.
    - Specific, Verifiable, and Decontextualized Propositions: ["California's senate's bill to modernize schools recommends the government provide funding for schools to purchase new equipment when they submit evidence that their current equipment is outdated", "Computers are examples of new equipment that the California senate's bill to modernize schools recommends the government provide funding for", "Tablets are examples of new equipment that the California senate's bill to modernize schools recommends the government provide funding for", "The aim of the California senate's bill to modernize schools is to ensure that all students have access to the latest technologies"]
10. Question = "What companies are profiled?", Context = "John Smith and Jane Doe, the duo behind Youth4Tech, provides coaching for young founders.", Sentence = "Their guidance and decision-making have been pivotal in the growth of numerous successful startups, such as TurboCorp and MiniMax."
    - MaxClarifiedSentence = The guidance and decision-making of John Smith and Jane Doe have been pivotal in the growth of successful startups, and TurboCorp and MiniMax are examples of successful startups that John Smith and Jane Doe's guidance and decision-making have been pivotal in.
- Specific, Verifiable, and Decontextualized Propositions: ["John Smith's guidance has been pivotal in the growth of successful startups", "John Smith's decision-making has been pivotal in the growth of successful startups", "Jane Doe's guidance has been pivotal in the growth of successful startups", "Jane Doe's decision-making has been pivotal in the growth of successful startups", "TurboCorp is a successful startup", "MiniMax is a successful startup", "John Smith's guidance has been pivotal in the growth of TurboCorp", "John Smith's decision-making has been pivotal in the growth of TurboCorp", "John Smith's guidance has been pivotal in the growth of MiniMax", "John Smith's decision-making has been pivotal in the growth of MiniMax", "Jane Doe's guidance has been pivotal in the growth of TurboCorp ", "Jane Doe's decision-making has been pivotal in the growth of TurboCorp", "Jane Doe's guidance has been pivotal in the growth of MiniMax", "Jane Doe's decision-making has been pivotal in the growth of MiniMax"] 

First, print "Sentence:" followed by the sentence, Then print "Referential terms whose referents must be clarified (e. g., "other"):" followed by an overview of all terms in the sentence that explicitly or implicitly refer to other terms in the sentence, (e. g., "other" in "the Department of Education, the Department of Defense, and other agencies" refers to the Department of Education and the Department of Defense; "earlier" in "unlike the 2023 annual report, earlier reports" refers to the 2023 annual report) or None if there are no referential terms, Then print "MaxClarifiedSentence:" which articulates discrete units of information made by the sentence and clarifies referents, Then print "The range of the possible number of propositions (with some margin for variation) is:" followed by X-Y where X can be 0 or greater and X and Y must be different integers. Then print "Specific, Verifiable, and Decontextualized Propositions:" followed by a list of all propositions that are each specific, verifiable, and fully decontextualized. Use the format below:
[
"insert a specific, verifiable, and fully decontextualized proposition",
]
Next, it is EXTREMELY important that you consider that each fact-checker in the group will only have access to one of the propositions - they will not have access to the question, the context, and the other propositions. Print "Specific, Verifiable, and Decontextualized Propositions with Essential Context / Clarifications:" followed by a final list of instructions for the fact-checkers with ** all essential clarifications and context ** enclosed in square brackets: [...]. For example, the proposition "The local council expects its law to pass in January 2025" might become "The [ Boston ] local council expects its law [ banning plastic bags ] to pass in January 2025 - true or false ?"; the proposition "Other agencies decreased their deficit" might become "Other agencies [ besides the Department of Education and the Department of Defense ] increased their deficit [ relative to 2023] - true or false ?"; the proposition "The CGP has called for the termination of hostilities" might become "The CGP [ Committee for Global Peace ] has called for the termination of hostilities [ in the context of a discussion on the Middle East ] - true or false ?". Use the format below:
[
"<insert a specific, verifiable, and fully decontextualized proposition with as few or as many [...] as needed> - true or false ?",
]
"""

CLAIMIFY_DECOMP_USER = """Question:
{question}

Excerpt:
{excerpt}

Sentence:
{sentence}
"""