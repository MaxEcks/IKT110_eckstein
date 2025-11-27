$(document).ready(function () {
    let radiant_pool = []
    let dire_pool = []

    $(".hero-select-left").click(function () {
        let hero_id = $(this).data("heroid");
        add_to_pool(radiant_pool, dire_pool, hero_id)
        update_images("radiant", radiant_pool)
    })

    $(".hero-select-right").click(function () {
        let hero_id = $(this).data("heroid");
        add_to_pool(dire_pool, radiant_pool, hero_id)
        update_images("dire", dire_pool)
    })

    $(".hero-pool-radiant").click(function () {
        let hero_id = $(this).data("heroid");
        if (hero_id !== -1) {
            remove_from_pool(radiant_pool, hero_id)
            refresh_images()
            update_images("radiant", radiant_pool)
            update_images("dire", dire_pool)
        }
    })

    $(".hero-pool-dire").click(function () {
        let hero_id = $(this).data("heroid");
        if (hero_id !== -1) {
            remove_from_pool(dire_pool, hero_id)
            refresh_images()
            update_images("dire", dire_pool)
            update_images("radiant", radiant_pool)
        }
    })

    let output_box = $("#outputOracle")

    $("#radiantPickBtn").click(function () { sendRequest("/radiant_pick") })
    $("#radiantBanBtn").click(function () { sendRequest("/radiant_ban") })
    $("#direPickBtn").click(function () { sendRequest("/dire_pick") })
    $("#direBanBtn").click(function () { sendRequest("/dire_ban") })

    // ===== Neu: Winner Prediction Button =====
    $("#predictWinnerBtn").click(function () {
    $.ajax({
        type: "POST",
        url: "/predict_winner",
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify({
            "radiant": radiant_pool,
            "dire": dire_pool
        }),
        dataType: "json",
        success: function (data) {
            let radiant_prob = (data.radiant * 100).toFixed(1) + "%"
            let dire_prob = (data.dire * 100).toFixed(1) + "%"
            output_box.val("Radiant: " + radiant_prob + "\nDire: " + dire_prob)
            setHeight(output_box)
        }
    });
})


    function sendRequest(url) {
        $.ajax({
            type: "POST",
            url: url,
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify({
                "radiant": radiant_pool,
                "dire": dire_pool
            }),
            dataType: "json",
            success: function (data) {
                let pretty_data = JSON.stringify(data, undefined, 4)
                output_box.val(pretty_data)
                setHeight(output_box)
            }
        })
    }

    function refresh_images() {
        $(".hero-pool").each(function () {
            $(this).attr("src", "/static/img/placeholder.png")
            $(this).data("heroid", -1);
        })
    }

    function update_images(faction, pool) {
        let imgs = $(`.hero-pool-${faction}`)
        for (const [i, hero_id] of pool.entries()) {
            $(imgs[i]).attr("src", `/static/img/avatar-sb/${hero_id}.png`)
            $(imgs[i]).css("height", "60px")
            $(imgs[i]).data("heroid", hero_id)
        }
    }

    function add_to_pool(pool, other_pool, id) {
        if (!pool.includes(id) && !other_pool.includes(id) && pool.length < 5) {
            pool.push(id)
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
