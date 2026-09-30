# Error examples (selected automatically)


## Seq2Seq + LSTM


### repeated

- **SRC:** or a young brother or sister who has athletic ability may find that recruiters try to entice him or her into a sports career .  
  **REF:** ወይም ደግሞ በአትሌቲክስ ዘርፍ ወጣቶችን የሚመለምሉ ሰዎች በዚህ ረገድ ጥሩ ችሎታ ያለውን አንድ ክርስቲያን ወጣት ወደ ስፖርቱ አለም እንዲገባ ሊያግባቡት ይሞክሩ ይሆናል።  
  **HYP:** ወይም አንድ ወንድም ወይም አንድ ወጣት ወይም ምስጢር ወይም በትምህርት ወይም በህብረተሰቡ ላይ ጉዳት ሊያስከትል ይችላል።
- **SRC:** yes , what spiritual riches jesus packed into his model prayer !  
  **REF:** በእርግጥም ኢየሱስ ያስተማረው የጸሎት ናሙና በርካታ መንፈሳዊ እንቁዎችን የያዘ ነው!  
  **HYP:** አዎን፣ ኢየሱስ፣ ኢየሱስ ወደ ሰማይ ጸልይ!
- **SRC:** these include the promised " new heavens and a new earth . "  
  **REF:** ይህም ቃል የተገባልንን " አዲስ ሰማይና አዲስ ምድር " ይጨምራል።  
  **HYP:** እነዚህ ቃላት " አዲስ ቃል ኪዳን " እና " አዲስ ኪዳን " ነው።

### missing

- **SRC:** besides , they claimed that they had neither the facilities nor the manpower to provide an alternative physical education program .  
  **REF:** ከዚህም በላይ አማራጭ የአካል ማጎልመሻ ትምህርት ለማዘጋጀት የሚረዱ መሳሪያዎችም ሆኑ የሰው ሀይል የለንም አለ።  
  **HYP:** ከዚህም በላይ ቲኦክራሲያዊ ትምህርት ቤቶችን የሚጠቀሙበት ጊዜ አለ።
- **SRC:** the dread of you makes my body tremble ;  
  **REF:** አንተን እጅግ ከመፍራቴ የተነሳ ሰውነቴ ይንቀጠቀጣል፤  
  **HYP:** እጆቻችሁን እንደ ገደለኝ፤
- **SRC:** as told by george warienchuck  
  **REF:** ጆርጅ ዎረንቸክ እንደተናገረው  
  **HYP:** ቲንደል፦

### additional

- **SRC:** obviously , the world ' s religions have not been immune from satan ' s influence .  
  **REF:** የአለም ሀይማኖቶች ከሰይጣን ተጽእኖ ነጻ አይደሉም።  
  **HYP:** በአለም ዙሪያ ያሉ ሀይማኖቶች በአለም ላይ ተጽእኖ የሚያሳድሩት ለምንድን ነው?
- **SRC:** my wife and i look at each other .  
  **REF:** እኔና ባለቤቴ ተያየን።  
  **HYP:** ባለቤቴና ባለቤቴ እርስ በርስ ተነጋገሩ።
- **SRC:** moreover , jehovah ' s goodness extends " to all . "  
  **REF:** ከዚህም በላይ ይሆዋ ጥሩነቱን የሚያሳየው " ለሁሉም " ነው።  
  **HYP:** ከዚህም በላይ ይሆዋ " መልካም የሆነውን ነገር ሁሉ [" NW] " በማለት ተናግሯል።

### word order

- **SRC:** hence , none of his servants need to be reluctant to accept help from those who are moved by jehovah to give such assistance .  
  **REF:** ይሆዋ አንዳንዶች ለሌሎች እርዳታ እንዲሰጡ ሊያነሳሳቸው ይችላል፤ በመሆኑም ከአገልጋዮቹ መካከል ማናቸውም ቢሆኑ እንዲህ ያለውን ድጋፍ ለመቀበል ማቅማማት የለባቸውም።  
  **HYP:** በመሆኑም ይሆዋ አገልጋዮቹን ለመርዳት የሚያስችላቸውን እርዳታ ለማግኘት ጥረት ማድረግ የሚችሉት እንዴት ነው?
- **SRC:** soon god ' s kingdom will be the only government ruling the earth .  
  **REF:** በቅርቡ በምድር ላይ የሚገዛው የአምላክ መንግስት ብቻ ይሆናል።  
  **HYP:** በቅርቡ የአምላክ መንግስት በቅርቡ በምድር ላይ ነው።
- **SRC:** many who have had the privilege of conducting progressive bible studies will tell you that few things are more rewarding .  
  **REF:** ጥሩ እድገት የሚያደርጉ የመጽሀፍ ቅዱስ ጥናቶችን የመምራት አጋጣሚ ያገኙ በርካታ ወንድሞችና እህቶች በዚህ ሀሳብ ይስማማሉ።  
  **HYP:** በርካታ የመጽሀፍ ቅዱስ ጥናቶችን በማገልገሉ በጣም ብዙ ሰዎች በጣም ብዙ ናቸው።

### named entity

- **SRC:** gaffar , who was born in turkey , was disturbed by the idea of a vengeful god , as taught by his religion .  
  **REF:** በቱርክ የተወለደው ጃፋር ሀይማኖቱ በሚያስተምረው ' አምላክ ተበቃይ ነው ' በሚለው ትምህርት ይረበሽ ነበር።  
  **HYP:** በ1940ዎቹ አመታት፣ ምስጢር፣ የስምንት ንዴት፣ ድምጾቹን በወንጌላዊነትና በህጻናዊነት ላይ የተመሰረተ ነበር።
- **SRC:** obviously , the world ' s religions have not been immune from satan ' s influence .  
  **REF:** የአለም ሀይማኖቶች ከሰይጣን ተጽእኖ ነጻ አይደሉም።  
  **HYP:** በአለም ዙሪያ ያሉ ሀይማኖቶች በአለም ላይ ተጽእኖ የሚያሳድሩት ለምንድን ነው?
- **SRC:** and moses and aaron went and gathered together all the elders of the children of israel :  
  **REF:** ሙሴና አሮንም ሄዱ የእስራኤልንም ልጆች ሽማግሌዎች ሁሉ ሰበሰቡ።  
  **HYP:** ሙሴም አሮንንና አሮንን እንዲህ አላቸው፥

### rare

- **SRC:** besides , they claimed that they had neither the facilities nor the manpower to provide an alternative physical education program .  
  **REF:** ከዚህም በላይ አማራጭ የአካል ማጎልመሻ ትምህርት ለማዘጋጀት የሚረዱ መሳሪያዎችም ሆኑ የሰው ሀይል የለንም አለ።  
  **HYP:** ከዚህም በላይ ቲኦክራሲያዊ ትምህርት ቤቶችን የሚጠቀሙበት ጊዜ አለ።
- **SRC:** as told by george warienchuck  
  **REF:** ጆርጅ ዎረንቸክ እንደተናገረው  
  **HYP:** ቲንደል፦
- **SRC:** or a young brother or sister who has athletic ability may find that recruiters try to entice him or her into a sports career .  
  **REF:** ወይም ደግሞ በአትሌቲክስ ዘርፍ ወጣቶችን የሚመለምሉ ሰዎች በዚህ ረገድ ጥሩ ችሎታ ያለውን አንድ ክርስቲያን ወጣት ወደ ስፖርቱ አለም እንዲገባ ሊያግባቡት ይሞክሩ ይሆናል።  
  **HYP:** ወይም አንድ ወንድም ወይም አንድ ወጣት ወይም ምስጢር ወይም በትምህርት ወይም በህብረተሰቡ ላይ ጉዳት ሊያስከትል ይችላል።

## Attention Seq2Seq + LSTM


### repeated

- **SRC:** " he ' s in heaven , " the commander in chief told the family of one fallen marine in a private moment .  
  **REF:** አንድ ከፍተኛ የሰራዊት አዛዥ በውጊያ ላይ ከሞተ አንድ የባህር ሀይል አባል ቤተሰብ ጋር ሲነጋገሩ " ገነት ገብቷል " ሲሉ መደመጣቸው ተሰማ።  
  **HYP:** " በሰማይ ያለው ንጉስ በሰማይ ያለው " በማለት ተናግሯል።
- **SRC:** or a young brother or sister who has athletic ability may find that recruiters try to entice him or her into a sports career .  
  **REF:** ወይም ደግሞ በአትሌቲክስ ዘርፍ ወጣቶችን የሚመለምሉ ሰዎች በዚህ ረገድ ጥሩ ችሎታ ያለውን አንድ ክርስቲያን ወጣት ወደ ስፖርቱ አለም እንዲገባ ሊያግባቡት ይሞክሩ ይሆናል።  
  **HYP:** ወይም ደግሞ አንድ ወጣት ወይም አንዲት ወጣት አንድ ወጣት ወደ ፖርቱጋል ሄድን።
- **SRC:** in recent decades , batik has gained greater popularity and has become a symbol of indonesian national identity .  
  **REF:** ብዙዎቹ የኢንዶኔዥያ ግዛቶች የራሳቸው የሆነ የአቀላለምና ንድፍ የማውጣት ዘዴ አላቸው።  
  **HYP:** ከቅርብ አስርተ አስርተ አመታት ወዲህ ምስጢራቸውን በጣም ውድ በሆነ መንገድ ተካፍሏል።

### missing

- **SRC:** the dread of you makes my body tremble ;  
  **REF:** አንተን እጅግ ከመፍራቴ የተነሳ ሰውነቴ ይንቀጠቀጣል፤  
  **HYP:** የህልም እንጀራዬን ትጠብቃለህ፤
- **SRC:** " he ' s in heaven , " the commander in chief told the family of one fallen marine in a private moment .  
  **REF:** አንድ ከፍተኛ የሰራዊት አዛዥ በውጊያ ላይ ከሞተ አንድ የባህር ሀይል አባል ቤተሰብ ጋር ሲነጋገሩ " ገነት ገብቷል " ሲሉ መደመጣቸው ተሰማ።  
  **HYP:** " በሰማይ ያለው ንጉስ በሰማይ ያለው " በማለት ተናግሯል።
- **SRC:** or a young brother or sister who has athletic ability may find that recruiters try to entice him or her into a sports career .  
  **REF:** ወይም ደግሞ በአትሌቲክስ ዘርፍ ወጣቶችን የሚመለምሉ ሰዎች በዚህ ረገድ ጥሩ ችሎታ ያለውን አንድ ክርስቲያን ወጣት ወደ ስፖርቱ አለም እንዲገባ ሊያግባቡት ይሞክሩ ይሆናል።  
  **HYP:** ወይም ደግሞ አንድ ወጣት ወይም አንዲት ወጣት አንድ ወጣት ወደ ፖርቱጋል ሄድን።

### additional

- **SRC:** as told by george warienchuck  
  **REF:** ጆርጅ ዎረንቸክ እንደተናገረው  
  **HYP:** በስፔይን ሪፑብሊክ ንገረን እንደተናገረው
- **SRC:** obviously , the world ' s religions have not been immune from satan ' s influence .  
  **REF:** የአለም ሀይማኖቶች ከሰይጣን ተጽእኖ ነጻ አይደሉም።  
  **HYP:** እርግጥ ነው፣ የአለም ሀይማኖቶች የሰይጣንን ተጽእኖ መቋቋም አልቻሉም።
- **SRC:** " he is not here , " said the angel , " for he was raised up . "  
  **REF:** አዎ፣ ኢየሱስ ህያው ሆኗል!  
  **HYP:** መልአኩ " እዚህ ላይ ተገለጠለት " በማለት መልአኩ ተናግሯል።

### word order

- **SRC:** these include the promised " new heavens and a new earth . "  
  **REF:** ይህም ቃል የተገባልንን " አዲስ ሰማይና አዲስ ምድር " ይጨምራል።  
  **HYP:** እነዚህ ሰዎች " አዲስ ሰማይና አዲስ ምድር " የሚለውን ቃል ያመለክታል።
- **SRC:** the answers to those questions will be discussed in the next article .  
  **REF:** የሚቀጥለው ርእስ የእነዚህን ጥያቄዎች መልስ ይዟል።  
  **HYP:** የእነዚህ ጥያቄዎች መልስ በሚቀጥለው ርእስ ላይ ይብራራል።
- **SRC:** only joshua and caleb urged the people not to rebel out of fear , for jehovah would surely be with them .  
  **REF:** ይሆዋ ከእነርሱ ጋር እንደሚሆን በመተማመን ህዝቡ ከፍርሀት የተነሳ ማመጽ እንደሌለባቸው የተናገሩት ኢያሱና ካሌብ ብቻ ነበሩ።  
  **HYP:** ኢያሱና ካሌብ፣ ይሆዋ ከእነሱ ጋር እንደሚሆን የረዳቸው ኢያሱ ብቻ ሳይሆን ካሌብና ካሌብ ብቻ ነው።

### named entity

- **SRC:** gaffar , who was born in turkey , was disturbed by the idea of a vengeful god , as taught by his religion .  
  **REF:** በቱርክ የተወለደው ጃፋር ሀይማኖቱ በሚያስተምረው ' አምላክ ተበቃይ ነው ' በሚለው ትምህርት ይረበሽ ነበር።  
  **HYP:** በቤቱ ውስጥ የተወለድኩበት ጊዜ አለ።
- **SRC:** by way of contrast , jehovah ' s witnesses endeavor to imitate jesus and his early disciples .  
  **REF:** ከዚህ በተቃራኒ የይሆዋ ምስክሮች ኢየሱስና የመጀመሪያው መቶ ዘመን ደቀመዛሙርት የተዉትን ምሳሌ ለመከተል ይጥራሉ።  
  **HYP:** በአንጻሩ ግን የይሆዋ ምስክሮችንና የጥንቶቹ ደቀ መዛሙርቱን ለመምሰል ይጥራሉ።
- **SRC:** and there remained among the children of israel seven tribes , which had not yet received their inheritance .  
  **REF:** ከእስራኤልም ልጆች ርስት ያልተካፈሉ ሰባት ነገድ ቀርተው ነበር።  
  **HYP:** ከሰባት ነገዶችም መካከል ርስታቸውን አልተቀበሉም።

### rare

- **SRC:** besides , they claimed that they had neither the facilities nor the manpower to provide an alternative physical education program .  
  **REF:** ከዚህም በላይ አማራጭ የአካል ማጎልመሻ ትምህርት ለማዘጋጀት የሚረዱ መሳሪያዎችም ሆኑ የሰው ሀይል የለንም አለ።  
  **HYP:** ከዚህም በላይ ምስጢሩና የሰው ልጆች የህክምና ትምህርት ቤቶችን ለማስተዳደር የሚያስችል አማራጭ የለም ብለው ይናገራሉ።
- **SRC:** as told by george warienchuck  
  **REF:** ጆርጅ ዎረንቸክ እንደተናገረው  
  **HYP:** በስፔይን ሪፑብሊክ ንገረን እንደተናገረው
- **SRC:** or a young brother or sister who has athletic ability may find that recruiters try to entice him or her into a sports career .  
  **REF:** ወይም ደግሞ በአትሌቲክስ ዘርፍ ወጣቶችን የሚመለምሉ ሰዎች በዚህ ረገድ ጥሩ ችሎታ ያለውን አንድ ክርስቲያን ወጣት ወደ ስፖርቱ አለም እንዲገባ ሊያግባቡት ይሞክሩ ይሆናል።  
  **HYP:** ወይም ደግሞ አንድ ወጣት ወይም አንዲት ወጣት አንድ ወጣት ወደ ፖርቱጋል ሄድን።
