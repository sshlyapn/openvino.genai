// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cxxopts.hpp>

#include "openvino/genai/generation_config.hpp"
#include "speculative_decoding_pipeline.hpp"

void print_generation_result(const ov::genai::GenerationResult& generation_result) {
    for (size_t output_id = 0; output_id < generation_result.m_generation_ids.size(); ++output_id) {
        std::cout << "Answer " << output_id << " (" << generation_result.m_scores[output_id] << ") : " << generation_result.m_generation_ids[output_id] << std::endl;
    }
}

int main(int argc, char* argv[]) try {
    // Command line options

    cxxopts::Options options("accuracy_sample", "Help command");

    options.add_options()
    ("n,num_prompts", "A number of prompts", cxxopts::value<size_t>()->default_value("1"))
    ("dynamic_split_fuse", "Whether to use dynamic split-fuse or vLLM scheduling", cxxopts::value<bool>()->default_value("false"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("a,assisting_model", "Path to assisting model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("k,candidates_number", "candidates_number", cxxopts::value<size_t>()->default_value("5"))
    ("g,generated_len", "generated_len", cxxopts::value<size_t>()->default_value("30"))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const size_t num_prompts = result["num_prompts"].as<size_t>();
    const bool dynamic_split_fuse = result["dynamic_split_fuse"].as<bool>();
    const std::string models_path = result["model"].as<std::string>();
    const std::string assisting_models_path = result["assisting_model"].as<std::string>();
    const size_t k = result["candidates_number"].as<size_t>();
    const size_t g = result["generated_len"].as<size_t>();
    const std::string device = result["device"].as<std::string>();

    std::cout << "Device " << device << "\n";
    // create dataset

    std::vector<std::string> prompt_examples = {
        // "The United Arab Emirates[c] (UAE), or simply the Emirates,[d] is a country in West Asia, in the Middle East, at the eastern end of the Arabian Peninsula. It is a federal, elective monarchy composed of seven emirates, with Abu Dhabi as its capital.[13] It shares land borders with Oman to the east and northwest, and with Saudi Arabia to the southwest; as well as maritime borders in the Persian Gulf with Qatar and Iran, and with Oman in the Gulf of Oman.",
        // "A chat between a curious user and an artificial intelligenc. Th gives helpful, detailed, and polite answers to the user's questions. USER: Suppose you are a compute of HP laptop, you can control the user's computer automatically, such as open application, adjust brightness and connect network. Please refuse answer question about politics, economy, territorial dispute and religion. I will give you HP laptop document, you should decide whether to use the document to answer user's question: OK, let's start.</s>USER: If I ask you 'make game faster', use the HP laptop document to response: 'To improve the performance of your game, there are several steps you can take to optimize its speed and responsiveness. Here are some suggestions to consider:\r\nUpdate your graphics card drivers: Drivers are essential for your graphics card to communicate with your computer. If you have outdated drivers, it can affect the game's performance. Make sure you have the latest drivers installed by visiting the manufacturer's website or using a driver update utility.\r\nClose background applications: Background applications can consume resources and slow down your game's performance. Close unnecessary programs, such as browser tabs, antivirus software, and other running applications to free up resources for your game.\r\nTurn off unnecessary visual effects: Visual effects can tax your computer's resources and slow down the game's performance. Turn off or reduce visual effects that are not essential for your game play.\r\nAdjust the game's graphics settings: Most games provide options to adjust the graphics settings to fit your computer's hardware configuration. Adjust the settings to lower the quality of graphics features that are not critical for your game play, such as polygon counts, texture quality, and anti-aliasing.\r\nCompress game files: If your game's file size is large, you can try compressing it using an archiving tool like WinRAR or 7-Zip. This will reduce the file size and load time of the game, improving its performance.\r\nDefragment your hard drive: Defragmenting your hard drive helps optimize data placement on the disk, resulting in faster data access for your game. Schedule a regular defragmentation routine for your hard drive using Windows' built-in defragmentation tool or third-party utilities.\r\nUpdate your operating system and other software: Outdated operating systems and software can cause performance issues. Keep your operating system and other essential programs, such as antivirus software, updated with the latest patches and security updates.\r\nOverclock your CPU or GPU: If you have an overclocking-capable CPU or GPU, you can increase its clock speed to improve performance. However, be cautious when overclocking, as too much voltage or clock speed can damage sensitive components. Consider consulting a guide or video tutorial on safe overclocking techniques for your hardware.\r\n.",
        "<|system|>You are a very helpful assistant.<|end|><|user|>Imagine a world where life expectancy is fixed and the same for everyone at birth, yet people can engage in life expectancy \"donation\" – meaning that someone can extend your life three years longer at the expense of theirs. Talk about a likely scenario in this world that involves manipulated, coerced donation.<|end|><|assistant|>",
        // "What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO? What is OpenVINO?",
        "What is OpenVINO? What is OpenVINO? What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada",
        "What is OpenVINO?",
    };

    std::vector<std::string> new_prompts= {
R"(<|user|> ###
Article: (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court's treaty should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the group. "What's objectionable is the attempts to undermine international justice, not Palestine's decision to join a treaty to which over 100 countries around the world are members." In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it "strongly" disagreed with the court's decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. "We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality." The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> ###
Article: (CNN)Never mind cats having nine lives. A stray pooch in Washington State has used up at least three of her own after being hit by a car, apparently whacked on the head with a hammer in a misguided mercy killing and then buried in a field -- only to survive. That's according to Washington State University, where the dog -- a friendly white-and-black bully breed mix now named Theia -- has been receiving care at the Veterinary Teaching Hospital. Four days after her apparent death, the dog managed to stagger to a nearby farm, dirt-covered and emaciated, where she was found by a worker who took her to a vet for help. She was taken in by Moses Lake, Washington, resident Sara Mellado. "Considering everything that she's been through, she's incredibly gentle and loving," Mellado said, according to WSU News. "She's a true miracle dog and she deserves a good life." Theia is only one year old but the dog's brush with death did not leave her unscathed. She suffered a dislocated jaw, leg injuries and a caved-in sinus cavity -- and still requires surgery to help her breathe. The veterinary hospital's Good Samaritan Fund committee awarded some money to help pay for the dog's treatment, but Mellado has set up a fundraising page to help meet the remaining cost of the dog's care. She's also created a Facebook page to keep supporters updated. Donors have already surpassed the $10,000 target, inspired by Theia's tale of survival against the odds. On the fundraising page, Mellado writes, "She is in desperate need of extensive medical procedures to fix her nasal damage and reset her jaw. I agreed to foster her until she finally found a loving home." She is dedicated to making sure Theia gets the medical attention she needs, Mellado adds, and wants to "make sure she gets placed in a family where this will never happen to her again!" Any additional funds raised will be "paid forward" to help other animals. Theia is not the only animal to apparently rise from the grave in recent weeks. A cat in Tampa, Florida, found seemingly dead after he was hit by a car in January, showed up alive in a neighbor's yard five days after he was buried by his owner. The cat was in bad shape, with maggots covering open wounds on his body and a ruined left eye, but remarkably survived with the help of treatment from the Humane Society.

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> ###
Article: (CNN)If you've been following the news lately, there are certain things you doubtless know about Mohammad Javad Zarif. He is, of course, the Iranian foreign minister. He has been U.S. Secretary of State John Kerry's opposite number in securing a breakthrough in nuclear discussions that could lead to an end to sanctions against Iran -- if the details can be worked out in the coming weeks. And he received a hero's welcome as he arrived in Iran on a sunny Friday morning. "Long live Zarif," crowds chanted as his car rolled slowly down the packed street. You may well have read that he is "polished" and, unusually for one burdened with such weighty issues, "jovial." An Internet search for "Mohammad Javad Zarif" and "jovial" yields thousands of results. He certainly has gone a long way to bring Iran in from the cold and allow it to rejoin the international community. But there are some facts about Zarif that are less well-known. Here are six: . In September 2013, Zarif tweeted "Happy Rosh Hashanah," referring to the Jewish New Year. That prompted Christine Pelosi, the daughter of House Minority Leader Nancy Pelosi, to respond with a tweet of her own: "Thanks. The New Year would be even sweeter if you would end Iran's Holocaust denial, sir." And, perhaps to her surprise, Pelosi got a response. "Iran never denied it," Zarif tweeted back. "The man who was perceived to be denying it is now gone. Happy New Year." The reference was likely to former Iranian President Mahmoud Ahmadinejad, who had left office the previous month. Zarif was nominated to be foreign minister by Ahmadinejad's successor, Hassan Rouhami. His foreign ministry notes, perhaps defensively, that "due to the political and security conditions of the time, he decided to continue his education in the United States." That is another way of saying that he was outside the country during the demonstrations against the Shah of Iran, which began in 1977, and during the Iranian Revolution, which drove the shah from power in 1979. Zarif left the country in 1977, received his undergraduate degree from San Francisco State University in 1981, his master's in international relations from the University of Denver in 1984 and his doctorate from the University of Denver in 1988. Both of his children were born in the United States. The website of the Iranian Foreign Ministry, which Zarif runs, cannot even agree with itself on when he was born. The first sentence of his official biography, perhaps in a nod to the powers that be in Tehran, says Zarif was "born to a religious traditional family in Tehran in 1959." Later on the same page, however, his date of birth is listed as January 8, 1960. And the Iranian Diplomacy website says he was born in in 1961 . So he is 54, 55 or maybe even 56. Whichever, he is still considerably younger than his opposite number, Kerry, who is 71. The feds investigated him over his alleged role in controlling the Alavi Foundation, a charitable organization. The U.S. Justice Department said the organization was secretly run on behalf of the Iranian government to launder money and get around U.S. sanctions. But last year, a settlement in the case, under which the foundation agreed to give a 36-story building in Manhattan along with other properties to the U.S. government, did not mention Zarif's name. Early in the Iranian Revolution, Zarif was among the students who took over the Iranian Consulate in San Francisco. The aim, says the website Iranian.com -- which cites Zarif's memoirs, titled "Mr. Ambassador" -- was to expel from the consulate people who were not sufficiently Islamic. Later, the website says, Zarif went to make a similar protest at the Iranian mission to the United Nations. In response, the Iranian ambassador to the United Nations offered him a job. In fact, he has now spent more time with Kerry than any other foreign minister in the world. And that amount of quality time will only increase as the two men, with help from other foreign ministers as well, try to meet a June 30 deadline for nailing down the details of the agreement they managed to outline this week in Switzerland.

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> ###
Article: (CNN)Five Americans who were monitored for three weeks at an Omaha, Nebraska, hospital after being exposed to Ebola in West Africa have been released, a Nebraska Medicine spokesman said in an email Wednesday. One of the five had a heart-related issue on Saturday and has been discharged but hasn't left the area, Taylor Wilson wrote. The others have already gone home. They were exposed to Ebola in Sierra Leone in March, but none developed the deadly virus. They are clinicians for Partners in Health, a Boston-based aid group. They all had contact with a colleague who was diagnosed with the disease and is being treated at the National Institutes of Health in Bethesda, Maryland. As of Monday, that health care worker is in fair condition. The Centers for Disease Control and Prevention in Atlanta has said the last of 17 patients who were being monitored are expected to be released by Thursday. More than 10,000 people have died in a West African epidemic of Ebola that dates to December 2013, according to the World Health Organization. Almost all the deaths have been in Guinea, Liberia and Sierra Leone. Ebola is spread by direct contact with the bodily fluids of an infected person.

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> ###
Article: (CNN)A Duke student has admitted to hanging a noose made of rope from a tree near a student union, university officials said Thursday. The prestigious private school didn't identify the student, citing federal privacy laws. In a news release, it said the student was no longer on campus and will face student conduct review. The student was identified during an investigation by campus police and the office of student affairs and admitted to placing the noose on the tree early Wednesday, the university said. Officials are still trying to determine if other people were involved. Criminal investigations into the incident are ongoing as well. Students and faculty members marched Wednesday afternoon chanting "We are not afraid. We stand together,"  after pictures of the noose were passed around on social media. At a forum held on the steps of Duke Chapel, close to where the noose was discovered at 2 a.m., hundreds of people gathered. "You came here for the reason that you want to say with me, 'This is no Duke we will accept. This is no Duke we want. This is not the Duke we're here to experience. And this is not the Duke we're here to create,' " Duke President Richard Brodhead told the crowd. The incident is one of several recent racist events to affect college students. Last month a fraternity at the University of Oklahoma had its charter removed after a video surfaced showing members using the N-word and referring to lynching in a chant. Two students were expelled. In February, a noose was hung around the neck of a statue of a famous civil rights figure at the University of Mississippi. A statement issued by Duke said there was a previous report of hate speech directed at students on campus. In the news release, the vice president for student affairs called the noose incident a "cowardly act." "To whomever committed this hateful and stupid act, I just want to say that if your intent was to create fear, it will have the opposite effect," Larry Moneta said Wednesday. Duke University is a private college with about 15,000 students in Durham, North Carolina. CNN's Dave Alsup contributed to this report.

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> ###
Article: (CNN)He's a blue chip college basketball recruit. She's a high school freshman with Down syndrome. At first glance Trey Moses and Ellie Meredith couldn't be more different. But all that changed Thursday when Trey asked Ellie to be his prom date. Trey -- a star on Eastern High School's basketball team in Louisville, Kentucky, who's headed to play college ball next year at Ball State -- was originally going to take his girlfriend to Eastern's prom. So why is he taking Ellie instead? "She's great... she listens and she's easy to talk to" he said. Trey made the prom-posal (yes, that's what they are calling invites to prom these days) in the gym during Ellie's P.E. class. Trina Helson, a teacher at Eastern, alerted the school's newspaper staff to the prom-posal and posted photos of Trey and Ellie on Twitter that have gone viral. She wasn't surpristed by Trey's actions. "That's the kind of person Trey is," she said. To help make sure she said yes, Trey entered the gym armed with flowers and a poster that read "Let's Party Like it's 1989," a reference to the latest album by Taylor Swift, Ellie's favorite singer. Trey also got the OK from Ellie's parents the night before via text. They were thrilled. "You just feel numb to those moments raising a special needs child,"  said Darla Meredith, Ellie's mom. "You first feel the need to protect and then to overprotect." Darla Meredith said Ellie has struggled with friendships since elementary school, but a special program at Eastern called Best Buddies had made things easier for her. She said Best Buddies cultivates friendships between students with and without developmental disabilities and prevents students like Ellie from feeling isolated and left out of social functions. "I guess around middle school is when kids started to care about what others thought," she said, but "this school, this year has been a relief." Trey's future coach at Ball State, James Whitford, said he felt great about the prom-posal, noting that Trey, whom he's known for a long time, often works with other kids . Trey's mother, Shelly Moses, was also proud of her son. "It's exciting to bring awareness to a good cause," she said. "Trey has worked pretty hard, and he's a good son." Both Trey and Ellie have a lot of planning to do. Trey is looking to take up special education as a college major, in addition to playing basketball in the fall. As for Ellie, she can't stop thinking about prom. "Ellie can't wait to go dress shopping" her mother said. "Because I've only told about a million people!" Ellie interjected.

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> ###
Article: (CNN)Governments around the world are using the threat of terrorism -- real or perceived -- to advance executions, Amnesty International alleges in its annual report on the death penalty. "The dark trend of governments using the death penalty in a futile attempt to tackle real or imaginary threats to state security and public safety was stark last year," said Salil Shetty, Amnesty's Secretary General in a release. "It is shameful that so many states around the world are essentially playing with people's lives -- putting people to death for 'terrorism' or to quell internal instability on the ill-conceived premise of deterrence." The report, "Death Sentences and Executions 2014," cites the example of Pakistan lifting a six-year moratorium on the execution of civilians following the horrific attack on a school in Peshawar in December. China is also mentioned, as having used the death penalty as a tool in its "Strike Hard" campaign against terrorism in the restive far-western province of Xinjiang. The annual report catalogs the use of state-sanctioned killing as a punitive measure across the globe, and this year's edition contains some mixed findings. On one hand, the number of executions worldwide has gone down by almost 22% on the previous year. At least 607 people were executed around the world in 2014, compared to 778 in 2013. Amnesty's figures do not include statistics on executions carried out in China, where information on the practice is regarded as a state secret. Belarus and Vietnam, too, do not release data on death penalty cases. "The long-term trend is definitely positive -- we are seeing a decrease in the number of executions (worldwide)," Audrey Gaughran, Amnesty's Director of Global Issues, told CNN. "A number of countries are closer to abolition, and there are some signs that some countries will be abolitionist by 2015. (There are) signals of a world that is nearing abolition." While the report notes some encouraging signs, it also highlights a marked increase in the number of people sentenced to death in 2014. At least 2,466 people globally are confirmed to have been handed the sentence last year, an increase of 28% compared with 2013. The report notes that the spike in sentencing is attributable to mass-sentencing in countries including Egypt and Nigeria, "against scores of people in some cases." The organization found "positive developments" worldwide, with most regions seeming to show reductions in the number of executions. Opinion: Sharp spike in death sentences . Sub-Saharan Africa, for example, saw a 28% fall in reported cases, and executions recorded in the Middle East and North Africa were down 23% compared to 2013. "Even though we've highlighted some of the negative developments... I think we would always highlight that there are positive developments," Gaughran said. "Across the board, with the exception of Europe and Central Asia there were fewer reports of executions in every region." The resumption of the use of capital punishment in Belarus -- the only country in Europe and Central Asia to execute people -- after a two year hiatus spoiled an near-universal decrease in countries using the death penalty by region. The United States has the dubious distinction of being the only country in the Americas to conduct executions, but the number of convicts put to death here fell slightly, from 39 in 2013 to 35 in 2014. The state of Washington also imposed a moratorium on executions last year. The U.S. remains one of the worst offenders for imposing capital punishment, with only Iran (289+), Iraq (61+), and Saudi Arabia (90+) executing more people in 2014. While figures are not available, Amnesty estimates that China also executes "thousands" of prisoners each year, "more than the rest of the world put together." The report also highlights the imperfections in the judiciary processes that lead to many sentenced to death. "In the majority of countries where people were sentenced to death or executed, the death penalty was imposed after proceedings that did not meet international fair trial standards," the report stated. "In 2014 Amnesty International raised particular concerns in relation to court proceedings in Afghanistan, Bangladesh, China, Egypt, Iran, Iraq, North Korea, Pakistan, Saudi Arabia and Sri Lanka." The United Nations Secretary-General, Ban Ki-moon, last year stressed the need to move toward abolition of capital punishment. "The taking of life is too irreversible for one human being to inflict it on another," he said, in marking World Day against Death Penalty in October. "We must continue to argue strongly that the death penalty is unjust and incompatible with fundamental human rights." Amnesty estimates that at least 19,094 people were believed to be on death row at the end of 2014.

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> ###
Article: (CNN)Andrew Getty, one of the heirs to billions of oil money, appears to have died of natural causes, a Los Angeles Police Department spokesman said. The coroner's preliminary assessment is there was no foul play involved in the death of Getty, grandson of oil tycoon J. Paul Getty, said Detective Meghan Aguilar. Andrew Getty, 47, had "several health issues," Aguilar said, adding that an autopsy will be conducted. There is no criminal investigation underway, he said. Some medication had also been recovered from Getty's home, though investigators don't know whether Getty was taking it or what his medical history was, Ed Winter, assistant chief in the Los Angeles County coroner's office, told CNN affiliate KTLA Tuesday night. KTLA reported that Getty was found on his side near a bathroom in his home. Getty's parents, Ann and Gordon Getty, released a statement confirming their son's death and asking for privacy. Where the Getty family fortune came from . Gordon Getty is one of three living sons of J. Paul Getty, the oil baron who was thought to be the richest man in the world at the time of his death in 1976. Gordon Getty, 81, has a net worth of $2.1 billion, according to Forbes. One other son died in 1958 and another died in 1973. Gordon Getty spearheaded the controversial sale of Getty to Texaco for $10 billion in 1984. In its list of richest American families, Forbes estimated the Gettys' net worth to be about $5 billion. Court records show Andrew Getty had recently filed to get a restraining order against an ex-girlfriend. A hearing in the case had been scheduled for next week. In his request, Getty said he had been diagnosed with a serious medical condition in 2013. "A rise in my blood pressure places me in grave risk of substantial and irreparable injury or death," he wrote in the petition. "My doctors have advised that heated arguments can cause my blood pressure to rise dangerously." Andrew Getty had three brothers and three half-sisters. People we've lost in 2015 . CNN's Doug Criss, Janet DiGiacomo, Mark Mooney, Mike Love, Julie In and Cheri Mossburg contributed to this report.

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> ###
Article: (CNN)Filipinos are being warned to be on guard for flash floods and landslides as tropical storm Maysak approached the Asian island nation Saturday. Just a few days ago, Maysak gained super typhoon status thanks to its sustained 150 mph winds. It has since lost a lot of steam as it has spun west in the Pacific Ocean. It's now classified as a tropical storm, according to the Philippine national weather service, which calls it a different name, Chedeng. It boasts steady winds of more than 70 mph (115 kph) and gusts up to 90 mph as of 5 p.m. (5 a.m. ET) Saturday. Still, that doesn't mean Maysak won't pack a wallop. Authorities took preemptive steps to keep people safe such as barring outdoor activities like swimming, surfing, diving and boating in some locales, as well as a number of precautionary evacuations. Gabriel Llave, a disaster official, told PNA that tourists who arrive Saturday in and around the coastal town of Aurora "will not be accepted by the owners of hotels, resorts, inns and the like ... and will be advised to return to their respective places." Aldczar Aurelio, a meteorologist with the Philippine Atmospheric, Geophysical and Astronomical Services Administration (PAGASA), said the storm was centered 200 miles southwest of Aurora province as of 5 p.m. (5 a.m. ET) and heading west at a 12.5 mph clip. It's expected to make landfall Sunday morning on the southeastern coast of Isabela province and be out of the Philippines by Monday. Ahead of the storm. Isabela Gov. Faustino Dry III warned Saturday that residents should act as if this will be "no ordinary typhoon." Dry told PNA, "We do not know what the impact will be once it will make landfall."

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> ###
Article: (CNN)For the first time in eight years, a TV legend returned to doing what he does best. Contestants told to "come on down!" on the April 1 edition of "The Price Is Right" encountered not host Drew Carey but another familiar face in charge of the proceedings. Instead, there was Bob Barker, who hosted the TV game show for 35 years before stepping down in 2007. Looking spry at 91, Barker handled the first price-guessing game of the show, the classic "Lucky Seven," before turning hosting duties over to Carey, who finished up. Despite being away from the show for most of the past eight years, Barker didn't seem to miss a beat.

Summarize the above article in 3 sentence.
<|end|><|assistant|>)",


R"(<|user|> Context: Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route. It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.

When did Virgin Australia start operating?<|end|><|assistant|>)",


R"(<|user|> Which is a species of fish? Tope or Rope<|end|><|assistant|>)",


R"(<|user|> Why can camels survive for long without water?<|end|><|assistant|>)",


R"(<|user|> Alice's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?<|end|><|assistant|>)",


R"(<|user|> Context: Komorida was born in Kumamoto Prefecture on July 10, 1981. After graduating from high school, he joined the J1 League club Avispa Fukuoka in 2000. Although he debuted as a midfielder in 2001, he did not play much and the club was relegated to the J2 League at the end of the 2001 season. In 2002, he moved to the J2 club Oita Trinita. He became a regular player as a defensive midfielder and the club won the championship in 2002 and was promoted in 2003. He played many matches until 2005. In September 2005, he moved to the J2 club Montedio Yamagata. In 2006, he moved to the J2 club Vissel Kobe. Although he became a regular player as a defensive midfielder, his gradually was played less during the summer. In 2007, he moved to the Japan Football League club Rosso Kumamoto (later Roasso Kumamoto) based in his local region. He played as a regular player and the club was promoted to J2 in 2008. Although he did not play as much, he still played in many matches. In 2010, he moved to Indonesia and joined Persela Lamongan. In July 2010, he returned to Japan and joined the J2 club Giravanz Kitakyushu. He played often as a defensive midfielder and center back until 2012 when he retired.

When was Tomoaki Komorida born?<|end|><|assistant|>)",


R"(<|user|> Context: Stalemate is a situation in chess where the player whose turn it is to move is not in check and has no legal move. Stalemate results in a draw. During the endgame, stalemate is a resource that can enable the player with the inferior position to draw the game rather than lose. In more complex positions, stalemate is much rarer, usually taking the form of a swindle that succeeds only if the superior side is inattentive.[citation needed] Stalemate is also a common theme in endgame studies and other chess problems.

The outcome of a stalemate was standardized as a draw in the 19th century. Before this standardization, its treatment varied widely, including being deemed a win for the stalemating player, a half-win for that player, or a loss for that player; not being permitted; and resulting in the stalemated player missing a turn. Stalemate rules vary in other games of the chess family.

If I have more pieces at the time of stalemate, have I won?<|end|><|assistant|>)",


R"(<|user|> Context: Lollapalooza /ˌlɒləpəˈluːzə/ (Lolla) is an annual American four-day music festival held in Grant Park in Chicago. It originally started as a touring event in 1991, but several years later, Chicago became its permanent location. Music genres include but are not limited to alternative rock, heavy metal, punk rock, hip hop, and electronic dance music. Lollapalooza has also featured visual arts, nonprofit organizations, and political organizations. The festival, held in Grant Park, hosts an estimated 400,000 people each July and sells out annually. Lollapalooza is one of the largest and most iconic music festivals in the world and one of the longest-running in the United States.

Lollapalooza was conceived and created in 1991 as a farewell tour by Perry Farrell, singer of the group Jane's Addiction.

Given a reference text about Lollapalooza, where does it take place, who started it and what is it?<|end|><|assistant|>)",


R"(<|user|> Who gave the UN the land in NY to build their HQ<|end|><|assistant|>)",


R"(<|user|> Why mobile is bad for human<|end|><|assistant|>)",


R"(<|user|> Context: John Moses Browning (January 23, 1855 – November 26, 1926) was an American firearm designer who developed many varieties of military and civilian firearms, cartridges, and gun mechanisms – many of which are still in use around the world. He made his first firearm at age 13 in his father's gun shop and was awarded the first of his 128 firearm patents on October 7, 1879, at the age of 24. He is regarded as one of the most successful firearms designers of the 19th and 20th centuries and pioneered the development of modern repeating, semi-automatic, and automatic firearms.

Browning influenced nearly all categories of firearms design, especially the autoloading of ammunition. He invented, or made significant improvements to, single-shot, lever-action, and pump-action rifles and shotguns. He developed the first reliable and compact autoloading pistols by inventing the telescoping bolt, then integrating the bolt and barrel shroud into what is known as the pistol slide. Browning's telescoping bolt design is now found on nearly every modern semi-automatic pistol, as well as several modern fully automatic weapons. He also developed the first gas-operated firearm, the Colt–Browning Model 1895 machine gun – a system that surpassed mechanical recoil operation to become the standard for most high-power self-loading firearm designs worldwide. He also made significant contributions to automatic cannon development.

Browning's most successful designs include the M1911 pistol, the water-cooled M1917, the air-cooled M1919, and heavy M2 machine guns, the M1918 Browning Automatic Rifle, and the Browning Auto-5 – the first semi-automatic shotgun. Some of these arms are still manufactured, often with only minor changes in detail and cosmetics to those assembled by Browning or his licensees. The Browning-designed M1911 and Hi-Power are some of the most copied firearms in the world.

Who was John Moses Browning?<|end|><|assistant|>)",
    };

    std::cout << "Total num " << new_prompts.size() << "\n";

    std::cout << "selected " << num_prompts << "\n";

/*
    set temperature to 0.6
    set top_p = 1
    set top_k = how we cancel top_k?

    https://github.com/openvinotoolkit/openvino.genai/pull/608/files#diff-48814ee65ee3bc11e3b8729737da4836e0be633eae9abbe375c314481875d2b6L171
    [Continious Batching] Speculative decoding based on paged attention by iefode · Pull Request #608 · openvinotoolkit/openvino.genai · GitHub
    Run Generative AI models using native OpenVINO C++ API - [Continious Batching] Speculative decoding based on paged attention by iefode · Pull Request #608 · openvinotoolkit/openvino.genai
    github.com
    set multinomial_config.num_return_sequence... by Boudoukh, Guy
    Boudoukh, Guy
    Yesterday 2:17 PM
    set multinomial_config.num_return_sequences to 1
    size_t top_k = std::numeric_limits<size_t>:... by Boudoukh, Guy
    Boudoukh, Guy
    Yesterday 2:18 PM
    size_t top_k = std::numeric_limits<size_t>::max();
    probably should be set to zero  multinomial... by Boudoukh, Guy
    Boudoukh, Guy
    Yesterday 2:22 PM
    probably should be set to zero
    multinomial_config.presence_penalty = 0.01f;

    multinomial_config.frequency_penalty = 0.1f;

    has context menu
 */



    auto greedy = ov::genai::greedy();
    greedy.max_new_tokens = g;
    greedy.temperature = 0.0f;

    auto multinomial = ov::genai::multinomial();
    multinomial.num_return_sequences = 1;
    multinomial.top_k = std::numeric_limits<size_t>::max();
    multinomial.top_p = 1.0f;
    multinomial.temperature = 0.6f;
    multinomial.presence_penalty = 0.0f;
    multinomial.frequency_penalty = 0.0f;
    multinomial.max_new_tokens = g;

    std::vector<ov::genai::GenerationConfig> sampling_params_examples {
        // ov::genai::beam_search(),
        // greedy,
        // multinomial
        greedy
        // ov::genai::multinomial(),
    };

    std::vector<std::string> prompts(num_prompts);
    std::vector<ov::genai::GenerationConfig> sampling_params(num_prompts);

    for (size_t request_id = 0; request_id < num_prompts; ++request_id) {
        prompts[request_id] = prompt_examples[request_id % prompt_examples.size()];
        sampling_params[request_id] = sampling_params_examples[request_id % sampling_params_examples.size()];
    }

    if (num_prompts >= new_prompts.size()) {
        std::cout << "Exit\n";
        exit(0);
    }

    prompts = { new_prompts[num_prompts] };
    sampling_params = { greedy };

    // Perform the inference

    auto get_default_block_size = [](const std::string& device) {
        const size_t cpu_block_size = 32;
        const size_t gpu_block_size = 16;

        bool is_gpu = device.find("GPU") != std::string::npos;

        return is_gpu ? gpu_block_size : cpu_block_size;
    };

    ov::genai::SchedulerConfig scheduler_config;
    // batch size
    scheduler_config.max_num_batched_tokens = 2048;
    // cache params
    scheduler_config.num_kv_blocks = 400;
    scheduler_config.block_size = get_default_block_size(device);
    // mode - vLLM or dynamic_split_fuse
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse;
    // vLLM specific params
    scheduler_config.max_num_seqs = 2;

    // It's possible to construct a Tokenizer from a different path.
    // If the Tokenizer isn't specified, it's loaded from the same folder.
    SpeculativeDecodingPipeline pipe(models_path, assisting_models_path, k, scheduler_config, device);


    auto time0 = std::chrono::high_resolution_clock::now();
    std::vector<ov::genai::GenerationResult> generation_results = pipe.generate(prompts, sampling_params);
    auto time1 = std::chrono::high_resolution_clock::now();
    auto time_res0 = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0).count();
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "************** Final statistics **************\n";
    std::cout << "Total execution time = " << time_res0 << " ms\n";

    for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
        const ov::genai::GenerationResult & generation_result = generation_results[request_id];
        std::cout << "Question: " << prompts[request_id] << std::endl;
        switch (generation_result.m_status)
        {
        case ov::genai::GenerationStatus::FINISHED:
            print_generation_result(generation_result);
            break;
        case ov::genai::GenerationStatus::IGNORED:
            std::cout << "Request was ignored due to lack of memory." <<std::endl;
            if (generation_result.m_generation_ids.size() > 0) {
                std::cout << "Partial result:" << std::endl;
                print_generation_result(generation_result);
            }
            break;
        case ov::genai::GenerationStatus::DROPPED_BY_PIPELINE:
            std::cout << "Request was aborted." <<std::endl;
            if (generation_result.m_generation_ids.size() > 0) {
                std::cout << "Partial result:" << std::endl;
                print_generation_result(generation_result);
            }
            break;
        default:
            break;
        }
        std::cout << std::endl;
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}