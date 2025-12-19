$(document).ready(function () {
    let radiant_pick_pool = []
    let dire_pick_pool = []
    let radiant_ban_pool = []
    let dire_ban_pool = []

    $(".hero-select-left").click(function () {
        let hero_id = $(this).data("heroid");
        add_to_pool(radiant_pick_pool, dire_pick_pool, hero_id)
        update_images("pool", "radiant", radiant_pick_pool)
    })

    $(".hero-select-right").click(function () {
        let hero_id = $(this).data("heroid");
        add_to_pool(dire_pick_pool, radiant_pick_pool, hero_id)
        update_images("pool", "dire", dire_pick_pool)
    })

    $(".hero-pool-radiant").click(function () {
        let hero_id = $(this).data("heroid");
        if (hero_id !== -1) {
            remove_from_pool(radiant_pick_pool, hero_id)
            refresh_images()
            update_images("pool", "radiant", radiant_pick_pool)
            update_images("pool", "dire", dire_pick_pool)
        }
    })

    $(".hero-pool-dire").click(function () {
        let hero_id = $(this).data("heroid");
        if (hero_id !== -1) {
            remove_from_pool(dire_pick_pool, hero_id)
            refresh_images()
            update_images("pool", "dire", dire_pick_pool)
            update_images("pool", "radiant", radiant_pick_pool)
        }
    })

    $(".hero-ban-radiant").click(function () {
        let hero_id = $(this).data("heroid");
        if (hero_id !== -1) {
            remove_from_pool(radiant_ban_pool, hero_id)
            refresh_images()
            update_images("ban", "radiant", radiant_ban_pool)
            update_images("ban", "dire", dire_ban_pool)
        }
    })

    $(".hero-ban-dire").click(function () {
        let hero_id = $(this).data("heroid");
        if (hero_id !== -1) {
            remove_from_pool(dire_ban_pool, hero_id)
            refresh_images()
            update_images("ban", "dire", dire_ban_pool)
            update_images("ban", "radiant", radiant_ban_pool)
        }
    })

    let output_box = $("#winner-result-box")
    
    function makeSuggestionsClickable() {
        $(".oracle-output-img").off("click").on("click", function() {
            let hero_id = $(this).data("heroid");
            if (hero_id === -1) return; // ignore empty boxes

            let isRadiant = $(this).css("border-color") === "rgb(0, 200, 83)"; // green
            if (isRadiant) {
                add_to_pool(radiant_pick_pool, dire_pick_pool, hero_id);
                update_images("pool", "radiant", radiant_pick_pool);
            } else { // Dire
                add_to_pool(dire_pick_pool, radiant_pick_pool, hero_id);
                update_images("pool", "dire", dire_pick_pool);
            }

            $(this).attr("src", "{{ url_for('static', filename='img/placeholder.png')}}");
            $(this).data("heroid", -1);
            $(this).css("border", "none");
        });

        $(".oracle-ban-output-img").off("click").on("click", function() {
            let hero_id = $(this).data("heroid");
            if (hero_id === -1) return;

            let isRadiant = $(this).css("border-color") === "rgb(0, 200, 83)";
            if (isRadiant) {
                add_to_pool(radiant_ban_pool, dire_ban_pool, hero_id, 7);
                update_images("ban", "radiant", radiant_ban_pool);
            } else { // Dire
                add_to_pool(dire_ban_pool, radiant_ban_pool, hero_id, 7);
                update_images("ban", "dire", dire_ban_pool);
            }

            $(this).attr("src", "{{ url_for('static', filename='img/placeholder.png')}}");
            $(this).data("heroid", -1);
            $(this).css("border", "none");
        });
    }

    // Call this function whenever suggestions are updated
    function showSuggestions(type, faction, recommended) {
        let prefix = type === "pick" ? "oracle-output-" : "oracle-ban-output-";

        for (let i = 0; i < 3; i++) {
            let hero_id = recommended[i] !== undefined ? parseInt(recommended[i]) : -1;
            let img = $(`#${prefix}${i}`);

            // Set image and data
            if (hero_id !== -1) {
                img.attr("src", `/static/img/avatar-sb/${hero_id}.png`);
                img.data("heroid", hero_id);
            } else {
                img.attr("src", "{{ url_for('static', filename='img/placeholder.png')}}");
                img.data("heroid", -1);
            }

            // Border color based on faction
            if (faction === "radiant") {
                img.css("border", "2px solid #00c853"); // green
            } else if (faction === "dire") {
                img.css("border", "2px solid #d50000"); // red
            }

            // Opacity for bans
            if (type === "ban") {
                img.css("opacity", "0.8");
            } else {
                img.css("opacity", "1");
            }

            // ===== REPLACE CLICK HANDLER HERE =====
            img.off("click").on("click", function() {
                let hero_id = $(this).data("heroid");
                if (hero_id === -1) return; // ignore placeholders

                if (type === "pick") {
                    if (faction === "radiant") {
                        add_to_pool(radiant_pick_pool, dire_pick_pool, hero_id);
                        update_images("pool", "radiant", radiant_pick_pool);
                    } else {
                        add_to_pool(dire_pick_pool, radiant_pick_pool, hero_id);
                        update_images("pool", "dire", dire_pick_pool);
                    }
                } else if (type === "ban") {  // <- use ban pools here
                    if (faction === "radiant") {
                        add_to_pool(radiant_ban_pool, dire_ban_pool, hero_id, 7);
                        update_images("ban", "radiant", radiant_ban_pool);
                    } else {
                        add_to_pool(dire_ban_pool, radiant_ban_pool, hero_id, 7);
                        update_images("ban", "dire", dire_ban_pool);
                    }
                }

                // Reset suggestion box after selection
                $(this).attr("src", "{{ url_for('static', filename='img/placeholder.png')}}");
                $(this).data("heroid", -1);
                $(this).css("border", "none");
            });
        }
    }
    $("#radiantPickBtn").click(function () {
        sendRequest("/radiant_pick", function(data) {
            showSuggestions("pick", "radiant", data.picks);
        });
    });

    $("#direPickBtn").click(function () {
        sendRequest("/dire_pick", function(data) {
            showSuggestions("pick", "dire", data.picks);
        });
    });

    // ===== Ban Buttons =====
    $("#radiantBanBtn").click(function () {
        sendRequest("/radiant_ban", function(data) {
            showSuggestions("ban", "radiant", data.bans);
        });
    });

    $("#direBanBtn").click(function () {
        sendRequest("/dire_ban", function(data) {
            showSuggestions("ban", "dire", data.bans);
        });
    });

    // ===== Neu: Winner Prediction Button =====
    $("#predictWinnerBtn").click(function () {
        // Pre-check picks
        if (radiant_pick_pool.length < 5 || dire_pick_pool.length < 5) {
            let resultBox = $("#winner-result-box");
            let resultText = $("#winner-result-text");
            resultText.text("Please select 5 heroes per team to predict the winner.");
            resultBox.css("border-left", "5px solid #ffb300"); // amber for warning
            resultBox.hide().fadeIn(150);
            return; // stop here, don't call server
        }

        // Only call server if both teams have 5 heroes
        $.ajax({
            type: "POST",
            url: "/predict_winner",
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({
                "radiant": radiant_pick_pool,
                "dire": dire_pick_pool
            }),
            dataType: "json",
            success: function (data) {
                let radiant_prob = (data.radiant * 100).toFixed(1) + "%";
                let dire_prob = (data.dire * 100).toFixed(1) + "%";

                let resultBox = $("#winner-result-box");
                let resultText = $("#winner-result-text");

                // Build result text
                let resultString = `Radiant: ${radiant_prob} | Dire: ${dire_prob}`;
                resultText.text(resultString);

                // Add color accent on the left side
                if (data.radiant > data.dire) {
                    resultBox.css("border-left", "5px solid #00c853"); // green
                } else if (data.radiant < data.dire) {
                    resultBox.css("border-left", "5px solid #d50000"); // red
                } else {
                    resultBox.css("border-left", "5px solid #9e9e9e"); // tie / gray
                }

                // Modern fade-in animation
                resultBox.hide().fadeIn(200);
            },
            error: function () {
                $("#winner-result-text").text("Prediction failed — try again.");
                $("#winner-result-box").css("border-left", "5px solid #d50000").hide().fadeIn(150);
            }
        });
    });

    function sendRequest(url, successCallback) {
        $.ajax({
            type: "POST",
            url: url,
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({
                radiant: radiant_pick_pool,
                dire: dire_pick_pool,
                radiant_ban: radiant_ban_pool,
                dire_ban: dire_ban_pool
            }),
            dataType: "json",
            success: function (data) {
                if (successCallback) successCallback(data); 
            },
            error: function(xhr, status, error) {
                console.error("Error:", error);
            }
        });
    }

    function refresh_images() {
        $(".hero-pool").each(function () {
            $(this).attr("src", "/static/img/placeholder.png")
            $(this).data("heroid", -1);
        })
    }

    function update_images(type, faction, pool) {
        let imgs = $(`.hero-${type}-${faction}`)
        for (const [i, hero_id] of pool.entries()) {
            $(imgs[i]).attr("src", `/static/img/avatar-sb/${hero_id}.png`)
            $(imgs[i]).css("height", "60px")
            $(imgs[i]).data("heroid", hero_id)
        }
    }

    function add_to_pool(pool, other_pool, id, max_length=5) {
        // List all pools to check
        let allPools = [radiant_pick_pool, dire_pick_pool, radiant_ban_pool, dire_ban_pool];

        // Check if hero exists in any pool
        let alreadyExists = allPools.some(p => p.includes(id));

        if (!alreadyExists && pool.length < max_length) {
            pool.push(id);
        }
    }

    function remove_from_pool(pool, id) {
        let idx = pool.indexOf(id)
        if (idx > -1) pool.splice(idx, 1)
    }

    function setHeight(jq_in){
        jq_in.each(function(index, elem){
            elem.style.height = elem.scrollHeight+'px'; 
        });
    }
});
