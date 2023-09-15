
 /* jQuery Pre loader
  -----------------------------------------------*/
$(window).load(function(){
    $('.preloader').fadeOut(1000); // set duration in brackets
});


/* Magnific Popup
-----------------------------------------------*/
$(document).ready(function() {
    $('.popup-youtube').magnificPopup({
        type: 'iframe',
        mainClass: 'mfp-fade',
        removalDelay: 160,
        preloader: false,
        fixedContentPos: false,
    });
});


/* Istope Portfolio
-----------------------------------------------*/
jQuery(document).ready(function($){

  if ( $('.iso-box-wrapper').length > 0 ) {

      var $container  = $('.iso-box-wrapper'),
        $imgs     = $('.iso-box img');

      $container.imagesLoaded(function () {

        $container.isotope({
        layoutMode: 'fitRows',
        itemSelector: '.iso-box'
        });

        $imgs.load(function(){
          $container.isotope('reLayout');
        });

      });

      //filter items on button click

      $('.filter-wrapper li a').click(function(){

          var $this = $(this), filterValue = $this.attr('data-filter');

      $container.isotope({
        filter: filterValue,
        animationOptions: {
            duration: 750,
            easing: 'linear',
            queue: false,
        }
      });

      // don't proceed if already selected

      if ( $this.hasClass('selected') ) {
        return false;
      }

      var filter_wrapper = $this.closest('.filter-wrapper');
      filter_wrapper.find('.selected').removeClass('selected');
      $this.addClass('selected');

        return false;
      });

  }

});


$(document).ready(function() {

  /* Hide mobile menu after clicking on a link
    -----------------------------------------------*/
    $('.navbar-collapse a').click(function(){
        $(".navbar-collapse").collapse('hide');
    });


  /*  smoothscroll
  ----------------------------------------------*/
   $(function() {
        $('#home a, .navbar-default a').bind('click', function(event) {
            var $anchor = $(this);
            $('html, body').stop().animate({
                scrollTop: $($anchor.attr('href')).offset().top - 49
            }, 1000);
            event.preventDefault();
        });
    });
    $(function() {
         $('#home a, .fa-angle-up a').bind('click', function(event) {
             var $anchor = $(this);
             $('html, body').stop().animate({
                 scrollTop: $($anchor.attr('href')).offset().top - 49
             }, 1000);
             event.preventDefault();
         });
     });



  /* home slideshow section
  -----------------------------------------------*/
  $(function(){
    jQuery(document).ready(function() {
    $('#home').backstretch([
      "images/392_0033.jpg",
       "images/IMG_5215.JPG",
       "images/PC300953.JPG",
       "images/IMG_5358.JPG",


     ],  {duration: 3000, fade: 1500});
    });
  });


   /* Flexslider
    -----------------------------------------------*/
    $(window).load(function() {
      $('.flexslider').flexslider({
         animation: "slide"
      });
    });


  /* Parallax section
    -----------------------------------------------*/
  function initParallax() {
    $('#home').parallax("100%", 0.1);
    $('#professional').parallax("100%", 0.1);
    $('#CVs').parallax("100%", 0.1);
    $('#links').parallax("100%", 10);
    $('#personal').parallax("100%", 0.1);
    $('#copyright').parallax("100%", 0.1);
  }
  initParallax();


  /* Nivo lightbox
    -----------------------------------------------*/
  $('#gallery .col-md-4 a').nivoLightbox({
        effect: 'fadeScale',
    });


  /* wow
  -------------------------------*/
  new WOW({ mobile: false }).init();

  });

/* Show more text */
function showDiv(TextID,ButtonID,BoxID,scrollBool) {

  if ($(TextID).css("display") == "block") {

        $(TextID).css("display","none");
        $(ButtonID).val("read more...");
        resizePastAll();
        // console.log("shrunk!");
        if (scrollBool) {
          $('html,body').animate({
            scrollTop: $(BoxID).parent().offset().top - $('#navbar').height()});
          }
        }
        else if ($(TextID).css("display") == "none") {

          $(TextID).css("display","block");
          $(ButtonID).val("show less...");
          resizePastAll();
        // console.log("expanded!");
        }
}

/* Align CV gray boxes */
function CVAlign(boxes,texts,headers) {
  var bottomPad = 10;
  var heights = texts.map(function(text){
    return $(text).height();
  });
  var tops = headers.map(function(header){
    return $(header).offset().top;
  });
  var max;
  if (Math.max(...tops) == Math.min(...tops)) { // if the boxes are rendered on the same line
    var headerHeights = headers.map(function(header){
      return $(header).height();
    });
    max = Math.max(...heights) + bottomPad;
    var headerDiff = headerHeights[0] - headerHeights[1]; //left header minus right header difference
    if (Math.max(heights[1], heights[2]) < (heights[0] + headerDiff)) { //if the CV box is the lowest, pad the right boxes
      $(boxes[0]).height(max);
      $(boxes[1]).height(max+headerDiff);
      $(boxes[2]).height(max+headerDiff);
    }
    else { //else, pad the CV box and make the right boxes the same size
      $(boxes[0]).height(max-headerDiff);
      $(boxes[1]).height(max);
      $(boxes[2]).height(max);
    }
  }
  else {
    max = Math.max(...heights.slice(1,)) + bottomPad;
    $(boxes[0]).height(heights[0] + bottomPad);
    $(boxes[1]).height(max);
    $(boxes[2]).height(max);
  }
}

/* Align professional box heights and button positions */
function professionalAlign(boxes,texts,buttons) {
  var bottomPad = 10;
  var collapsedHeight = texts.map(function(text){
    return $(text).parent().children('div.snippet').height();
  });
  var maxCollapsedHeight = Math.max(...collapsedHeight);
  var bottomOffset = bottomPad + 4 * $(buttons[0]).height();
  var expandedHeight = texts.map(function(text){
    if ($(text).css("display") == "block") {
      return $(text).height() + $(text).parent().children('div.snippet').height();
    }
    else { return 0; }
  });
  var maxExpandedHeight = Math.max(...expandedHeight);
  for (var b = 0; b < boxes.length; b++) {
    var leftAlign = $(boxes[b]).outerWidth() / 2 - $(buttons[b]).outerWidth() / 2;
    $(buttons[b]).css("left", leftAlign);
    if ($(boxes[b]).children('div.moreText').css("display") == "none") { // if it's not expanded
      $(buttons[b]).css("bottom", 2 * bottomPad + "px");
      $(boxes[b]).height(maxCollapsedHeight + bottomOffset);
    }
    else if ($(boxes[b]).children('div.moreText').css("display") == "block") { // if it's expanded
      $(buttons[b]).css("bottom", "0px");
      $(boxes[b]).height(maxExpandedHeight + bottomOffset);
    }
  }
}

/* Align professional snippets to line up despite header heights */
function alignSnippets(abstracts, boxes){
  var boxTops = boxes.map(function(box){
    return $(box).offset().top;
  });
  var alignAbstracts = function(abstracts){
    var tops = abstracts.map(function(abstract){
      return $(abstract).offset().top;
    });
    var gaps = tops.map(function(top,_,tops){
      return Math.max(...tops) - top;
    });
    for (var a = 0; a < abstracts.length; a++) {
      $(abstracts[a]).css("padding-top", gaps[a]);
    }
  };
  if (Math.max(...boxTops) == Math.min(...boxTops)) {
    alignAbstracts(abstracts);
  }
  else if (boxTops[0] == boxTops[1]) {
    alignAbstracts(abstracts.slice(0, 2));
    $(abstracts[2]).css("padding-top", 0);
  }
  else {
    for (var a = 0; a < abstracts.length; a++) {
      $(abstracts[a]).css("padding-top", 0);
    }
  }
}

/* Function for realigning everything on window resize */
function resizeAll() {
  CVAlign(['#resumeBox','#recentlyBox','#soonBox'],['#resumeText','#recentlyText','#soonText'],['#CVHeader','#happeningsHeader']);
}

function resizePastAll() {
  alignSnippets(['#abstract1', '#abstract2', '#abstract3'], ['#pastProText1', '#pastProText2', '#pastProText3']);
  alignSnippets(['#abstract4', '#abstract5', '#abstract6'], ['#pastProText4', '#pastProText5', '#pastProText6']);
  professionalAlign(['#pastProBox1', '#pastProBox2', '#pastProBox3'], ["#moreNLP", "#moreDSP", "#moreProgramming"], ['#NLPButton', '#DSPButton', '#programmingButton']);
  professionalAlign(['#pastProBox4', '#pastProBox5', '#pastProBox6'], ["#moreFairness", "#moreSPiN", "#morePast"], ['#fairButton', '#SPiNButton', '#pastButton']);
}
