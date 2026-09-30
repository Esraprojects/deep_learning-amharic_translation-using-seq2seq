# Error examples (selected automatically)


## Seq2Seq + LSTM


### repeated

- **SRC:** as you engage in this work , you have the opportunity to " save both yourself and those who listen to you . "  
  **REF:** በዚህ ስራ ላይ መካፈልህ ' ራስህንና የሚሰሙህን ለማዳን ' የሚያስችል አጋጣሚ ያስገኝልሀል።  
  **HYP:** በዚህ መንገድ " [] " ["] " የሚለውን ቃል ምክሩን ቀጥሉ።
- **SRC:** i truly feel like the psalmist who sang : " blessed be jehovah , who daily carries the load for us . "  
  **REF:** እንደሚከተለው ሲል የዘመረው መዝሙራዊ አይነት ስሜት ይሰማኛል: - " በየቀኑ ሸክማችንን የሚሸከምልን ይሆዋ የተባረከ ይሁን። "  
  **HYP:** መዝሙራዊው " ይሆዋ ሆይ፣ ["] " ስለ ይሆዋ ያስባል፤ ምክንያቱም ስለ ይሆዋ ያስባል።
- **SRC:** he came to love the god he learned about , and that knowledge helped him to build faith .  
  **REF:** ስለ አምላክ የተማረው ነገር እሱን እንዲወደው ያደረገ ሲሆን እውቀቱ ደግሞ እምነት እንዲያዳብር ረድቶታል።  
  **HYP:** አምላክ ስለ እሱ እውነቱን ማወቅና ስለ እሱ እውነቱን ለማወቅ የሚያስችል አጋጣሚ ሰጥቷል።

### missing

- **SRC:** and as it was in the days of noe , so shall it be also in the days of the son of man .  
  **REF:** በኖህ ዘመንም እንደ ሆነ፥ በሰው ልጅ ዘመን ደግሞ እንዲሁ ይሆናል።  
  **HYP:** በንጉስም ዘመን በህልምና በስሜት ላይ ነው።
- **SRC:** so they differed with one another in their task , and secretly conferred .  
  **REF:** (ድግምተኞቹ) በመካከላቸውም ነገራቸውን ተጨቃጨቁ። ውይይትንም ደበቁ።  
  **HYP:** ስለዚህም በወንዶች ላይ ሰፈሩ።
- **SRC:** " when our son called and told me about a shooting at school , i couldn ' t believe it , " recalls heike .  
  **REF:** ሄይከ የተባለች አንዲት ሴት እንዲህ ብላለች፦ " ልጃችን ደውሎ በትምህርት ቤት ውስጥ እየተፈጸመ ስላለው ግድያ ሲነግረኝ ማመን አልቻልኩም።  
  **HYP:** " ቤተሰቦቼን ስለ ምስጢን ምግባራችንን ውደድ " በማለት ተናግሯል።

### additional

- **SRC:** my wife and i were violent toward each other , largely because we harbored feelings of jealousy .  
  **REF:** እኔና ባለቤቴ ስለምንቀናና ብዙ ጊዜ እንጣላ ነበር።  
  **HYP:** እኔና ባለቤቴና ጎረቤቶቻችንን ከወንዶች ጋር በተያያዘ በምስጢር ላይ ተጽእኖ ሊያሳድር ይችላል።
- **SRC:** you will not find any bible text that uses the expression " immortal soul "  
  **REF:** መጽሀፍ ቅዱስ ውስጥ " የማትሞት ነፍስ " የሚል አገላለጽ አይገኝም  
  **HYP:** መጽሀፍ ቅዱስ " ነፍስ " የሚለው ቃል " ነፍስ " የሚለው ቃል " ነፍስ "
- **SRC:** it is attainable .  
  **REF:** ይህንን ማድረግ ይቻላል።  
  **HYP:** ይህ ደግሞ በጣም አስፈላጊ ነው።

### word order

- **SRC:** if someone receives a theocratic assignment or a spiritual blessing , others in the congregation need to guard against envy .  
  **REF:** አንድ ሰው ቲኦክራሲያዊ ስራ ወይም መንፈሳዊ በረከት ሲያገኝ ሌሎች በጉባኤው ውስጥ ያሉ ሰዎች እንዳይቀኑ መጠንቀቅ ያስፈልጋቸዋል።  
  **HYP:** አንድ ክርስቲያን በጉባኤ ውስጥ ያሉ መንፈሳዊ ግቦችን የሚጠብቁ ከሆነ በመንፈሳዊ ወይም በመንፈሳዊ እንቅስቃሴዎች ላይ ተጽእኖ ሊያሳድር ይችላል።
- **SRC:** furthermore , they must be baptized christians who are " born again , " begotten by god ' s holy spirit .  
  **REF:** ከዚህም በላይ ደግሞ በአምላክ ቅዱስ መንፈስ ' ዳግመኛ የተወለዱ ' የተጠመቁ ክርስቲያኖች መሆን አለባቸው።  
  **HYP:** ከዚህም በተጨማሪ ክርስቲያኖች ' የአምላክ መንፈስ ' ማለትም በመንፈስ ቅዱስ ውስጥ እንዲገቡ ይረዳቸዋል።
- **SRC:** the result often is that one will procrastinate in making a decision , putting things off until it is too late .  
  **REF:** ብዙውን ጊዜ አንድ ሰው ውሳኔ ከማድረግ ይልቅ ዛሬ ነገ እያለ አንድን ጉዳይ ማጓተቱ ወደ ባሰ ችግር ውስጥ እንዲገባ ሊያደርገው ይችላል።  
  **HYP:** ይህ ደግሞ አንድ ሰው በምግብበት ጊዜ ላይ የሚያጋጥምው ነገር ነው።

### named entity

- **SRC:** furthermore , satan ' s earthly agents have persecuted servants of god to the point of death , even as they did jesus .  
  **REF:** ይህ መሆኑ ግን ሰይጣን የፈለገውን ሁሉ ለመግደል የሚያስችል ሀይል እንዳለው አያመለክትም።  
  **HYP:** ከዚህም በተጨማሪ ኢየሱስ ክርስቶስ የአምላክ መንግስት፣ የሰው ልጆችን ጨምሮ ሰዎች፣ የዲያብሎስን ህይወት እንዲያስወግድ አድርጓል።
- **SRC:** israel ' s older men initially believed moses and aaron .  
  **REF:** የእስራኤል ሽማግሌዎች መጀመሪያ ላይ ሙሴና አሮንን አምነው ተቀበሏቸው።  
  **HYP:** የእስራኤልም ልጆችም በያእቆብም ላይ ይከራከሩ ነበር።
- **SRC:** when adam and eve rebelled against god , they adopted the standards of the selfish traitor satan and chose him as their spiritual father .  
  **REF:** አዳምና ሄዋን በአምላክ ላይ ሲያምጹ ራስ ወዳድ የሆነውን የከሀዲውን የሰይጣንን መስፈርቶች በመከተል እርሱ መንፈሳዊ አባታቸው እንዲሆን መርጠዋል።  
  **HYP:** አዳምና ሄዋንን ጨምሮ ይሆዋ አምላክ የሰጣቸውን መንፈሳዊና መንፈሳዊ ምግብ እንዲሰጣቸውና እንዲታዘዙት ይፈልጋል።

### rare

- **SRC:** and as it was in the days of noe , so shall it be also in the days of the son of man .  
  **REF:** በኖህ ዘመንም እንደ ሆነ፥ በሰው ልጅ ዘመን ደግሞ እንዲሁ ይሆናል።  
  **HYP:** በንጉስም ዘመን በህልምና በስሜት ላይ ነው።
- **SRC:** " when our son called and told me about a shooting at school , i couldn ' t believe it , " recalls heike .  
  **REF:** ሄይከ የተባለች አንዲት ሴት እንዲህ ብላለች፦ " ልጃችን ደውሎ በትምህርት ቤት ውስጥ እየተፈጸመ ስላለው ግድያ ሲነግረኝ ማመን አልቻልኩም።  
  **HYP:** " ቤተሰቦቼን ስለ ምስጢን ምግባራችንን ውደድ " በማለት ተናግሯል።
- **SRC:** i was born in ilocos norte , philippines , on december 10 , 1968 , the seventh of ten children .  
  **REF:** ታህሳስ 10, 1968 በኢሎኮስ ኖርቲ፣ ፊሊፒንስ ተወለድኩ፤ በቤተሰባችን ውስጥ ካሉት አስር ልጆች መካከል እኔ ሰባተኛ ነበርኩ።  
  **HYP:** መስከረም 1966 180 አመት ሲሆነኝ 18 አመት ልጅ ሳለሁ ሬንስስ።

## Attention Seq2Seq + LSTM


### repeated

- **SRC:** my conscience troubled me more and more .  
  **REF:** በዚህ ጊዜ ለውትድርና ተጠራሁ።  
  **HYP:** ህሊናዬ ይበልጥ ይበልጥ ተነካኝ።
- **SRC:** and as it was in the days of noe , so shall it be also in the days of the son of man .  
  **REF:** በኖህ ዘመንም እንደ ሆነ፥ በሰው ልጅ ዘመን ደግሞ እንዲሁ ይሆናል።  
  **HYP:** በምእራም ዘመንም ሁሉ በህልም ዘመን፥ በወንጌልም ዘመን፥
- **SRC:** so they differed with one another in their task , and secretly conferred .  
  **REF:** (ድግምተኞቹ) በመካከላቸውም ነገራቸውን ተጨቃጨቁ። ውይይትንም ደበቁ።  
  **HYP:** በ1950ዎቹ አመታት በ1950ዎቹ አመታት ሰፈሩ።

### missing

- **SRC:** so they differed with one another in their task , and secretly conferred .  
  **REF:** (ድግምተኞቹ) በመካከላቸውም ነገራቸውን ተጨቃጨቁ። ውይይትንም ደበቁ።  
  **HYP:** በ1950ዎቹ አመታት በ1950ዎቹ አመታት ሰፈሩ።
- **SRC:** they decided to disobey god .  
  **REF:** አዳምና ሄዋን ይሆዋን ላለመታዘዝ ወሰኑ።  
  **HYP:** አምላክን መፍራት ጀመሩ።
- **SRC:** to whom should youngsters turn to find accurate knowledge that will safeguard them ?  
  **REF:** ወጣቶች ከአደጋ የሚጠብቃቸውን ትክክለኛ እውቀት ለማግኘት ወደ ማን ዞር ማለት አለባቸው?  
  **HYP:** ሰዎች የሚያጋጥሟቸውን እውቀት ማወቅ ያለባቸው እነማን ናቸው?

### additional

- **SRC:** notify a confidant , and if possible , call him before each dose is taken .  
  **REF:** እያንዳንዱን የምትወስደውን መድሀኒት መጠን መዝግበህ ያዝ።  
  **HYP:** ምእራብና ቢሻም (ቢል)። ከወንዶችም በፊት ከፊተኞቹም (ከቁርአን) አቀናው።
- **SRC:** ( the new testament of our lord jesus christ , translated from greek by reijnier rooleeuw , m . d . )  
  **REF:** (የጌታችን የኢየሱስ ክርስቶስ አዲስ ኪዳን በሬኒር ሮሌኦ ኤም ዲ ከግሪክኛ የተተረጎመ)  
  **HYP:** (ኒውንያ 19ን ትርጉም) " የጌታችን " የሚለው ስም ከጻፍነው የእብራይስጥ ክርስቶስ ጋር በተያያዘ ነው።
- **SRC:** my wife and i were violent toward each other , largely because we harbored feelings of jealousy .  
  **REF:** እኔና ባለቤቴ ስለምንቀናና ብዙ ጊዜ እንጣላ ነበር።  
  **HYP:** ባለቤቴም ሆነ እኔ ደግሞ ቅናትን በመቀበል ምክንያት ምኞቴን ጠላታችንን ተቀበልኩ።

### word order

- **SRC:** you will not find any bible text that uses the expression " immortal soul "  
  **REF:** መጽሀፍ ቅዱስ ውስጥ " የማትሞት ነፍስ " የሚል አገላለጽ አይገኝም  
  **HYP:** " ሞት " የሚለው አገላለጽ " ሞት " የሚለውን ቃል መጽሀፍ ቅዱስ ይናገራል
- **SRC:** contrary to popular belief , men do not always remarry simply to satisfy their physical or sexual needs .  
  **REF:** ብዙዎች ካላቸው አስተሳሰብ በተቃራኒ ወንዶች ብዙውን ጊዜ እንደገና የሚያገቡት አካላዊ ፍላጎታቸውን ወይም የጾታ ስሜታቸውን ለማርካት አይደለም።  
  **HYP:** ሰዎች እንዲህ ያለ እምነት ያላቸው ሰዎች ወይም የጾታ ፍላጎት ያላቸው ሰዎች አካላዊ ፍላጎት እንዳላቸው ይሰማቸዋል።
- **SRC:** during his 18 years in padua , three children were born to galileo by his mistress , a young venetian woman .  
  **REF:** በፓዱዋ በኖረባቸው 18 አመታት ቁባቱ ከነበረች ቬኒሲያዊት ሴት ሶስት ልጆች ወልዷል።  
  **HYP:** በ18 አመት ውስጥ ሶስት ልጆች በ18 አመት እድሜ ላይ ሲገባ የሴት ሴት ልጅ ወለድን።

### named entity

- **SRC:** ( the new testament of our lord jesus christ , translated from greek by reijnier rooleeuw , m . d . )  
  **REF:** (የጌታችን የኢየሱስ ክርስቶስ አዲስ ኪዳን በሬኒር ሮሌኦ ኤም ዲ ከግሪክኛ የተተረጎመ)  
  **HYP:** (ኒውንያ 19ን ትርጉም) " የጌታችን " የሚለው ስም ከጻፍነው የእብራይስጥ ክርስቶስ ጋር በተያያዘ ነው።
- **SRC:** this touching psalm of david should motivate us to be courageous and optimistic .  
  **REF:** ይህ ልብን የሚነካ የዳዊት መዝሙር ደፋሮችና ብሩህ የሆነው ጎን የሚታየን እንድንሆን ሊያነሳሳን ይገባል።  
  **HYP:** ይህ መዝሙር ደፋርና ምክንያታዊ እንድንሆን የሚረዳን እንዴት ነው?
- **SRC:** quoting jesus , the disciple mark wrote that this work must be done " first , " that is , before the end comes .  
  **REF:** ደቀ መዝሙሩ ማርቆስ፣ ኢየሱስ የተናገረውን በመጥቀስ ይህ ስራ " አስቀድሞ " ይኸውም መጨረሻው ከመምጣቱ በፊት መሰራት እንዳለበት ጽፏል።  
  **HYP:** ደቀ መዛሙርቱ ይህን ስራ ከመፈጸም በፊት " በመጀመሪያ " የሚለው ስራ " በመጀመሪያ " መሆን አለበት።

### rare

- **SRC:** and as it was in the days of noe , so shall it be also in the days of the son of man .  
  **REF:** በኖህ ዘመንም እንደ ሆነ፥ በሰው ልጅ ዘመን ደግሞ እንዲሁ ይሆናል።  
  **HYP:** በምእራም ዘመንም ሁሉ በህልም ዘመን፥ በወንጌልም ዘመን፥
- **SRC:** " when our son called and told me about a shooting at school , i couldn ' t believe it , " recalls heike .  
  **REF:** ሄይከ የተባለች አንዲት ሴት እንዲህ ብላለች፦ " ልጃችን ደውሎ በትምህርት ቤት ውስጥ እየተፈጸመ ስላለው ግድያ ሲነግረኝ ማመን አልቻልኩም።  
  **HYP:** " ልጃችንን ሲጠራው፣ በትምህርት ቤት ውስጥ ስለ አንድ ትንሽ አገር ስለ ምስቡም አልነበኝም " በማለት ተናግሯል።
- **SRC:** i was born in ilocos norte , philippines , on december 10 , 1968 , the seventh of ten children .  
  **REF:** ታህሳስ 10, 1968 በኢሎኮስ ኖርቲ፣ ፊሊፒንስ ተወለድኩ፤ በቤተሰባችን ውስጥ ካሉት አስር ልጆች መካከል እኔ ሰባተኛ ነበርኩ።  
  **HYP:** በታህሳስ 10, 1968 የ1968 19 አመት ልጅ ሳለሁ የሴት ልጆችና ካርድ ነበር።
